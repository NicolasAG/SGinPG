import argparse
import yaml
import copy
from sacremoses import MosesTokenizer, MosesDetokenizer

tokenizer = MosesTokenizer(lang='en')
detokenizer = MosesDetokenizer(lang='en')


parser = argparse.ArgumentParser()
parser.add_argument("--truth", type=str, required=True,
                    help="path to the ground truth file")
parser.add_argument("--pred", type=str, required=True,
                    help="path to the generated file")
parser.add_argument("--proof_type", type=str, default='normal',
                    help="proof type: 'normal', 'reversed'")


def tokenize(line):
    line = tokenizer.tokenize(line.strip(), return_str=True, escape=False)
    line = line.replace("e _ 1", "e_1")
    line = line.replace("e _ 2", "e_2")
    return line


def inverse_e1_and_e2(phrase):
    # inverse e_1 and e_2
    phrase = phrase.replace('e_1', 'e_tmp')
    phrase = phrase.replace('e_2', 'e_1')
    phrase = phrase.replace('e_tmp', 'e_2')
    return phrase


def get_relation_mappings():
    """
    :param relations: yaml dict of relation to partial phrases
    :return: a mapping from relation to all possible phrases
           & a mapping from gender aware relation to generic relation
    """
    rel2phrases = {}
    grel2rel = {}
    for key, val in relations.items():
        for gender in ['male', 'female']:
            rel = val[gender]['rel']
            if gender == 'male':
                other_gender = 'female'
            else:
                other_gender = 'male'

            rel2phrases[rel] = []
            grel2rel[rel] = key

            # add original phrases
            for p in val[gender]['p']:
                rel2phrases[rel].append(tokenize(p))

            # check if inverse relation exists
            if 'inv-' in key:
                other_key = key.replace('inv-', '')
            elif 'inv-' + key in relations:
                other_key = 'inv-' + key
            else:
                other_key = None
            if other_key:
                # add those phrases as well
                for p in relations[other_key][gender]['p']:
                    rel2phrases[rel].append(tokenize(inverse_e1_and_e2(p)))
                for p in relations[other_key][other_gender]['p']:
                    rel2phrases[rel].append(tokenize(inverse_e1_and_e2(p)))

            # special case for husband-wife relations:
            # the opposite is the gender
            if key == 'SO':
                for p in val[other_gender]['p']:
                    rel2phrases[rel].append(tokenize(inverse_e1_and_e2(p)))

            # special case for siblings:
            # the opposite is either the gender or the entity
            elif 'sibling' in key:
                for p in val[gender]['p'] + val[other_gender]['p']:
                    rel2phrases[rel].append(tokenize(inverse_e1_and_e2(p)))

    # extra hack for neice -vs- niece
    # accept both syntaxes due to vocab error
    rel2phrases['neice'] = copy.copy(rel2phrases['niece'])
    for idx, phrase in enumerate(rel2phrases['neice']):
        rel2phrases['neice'][idx] = phrase.replace('niece', 'neice')

    return rel2phrases, grel2rel


def regularize(line):
    return line.replace("Since ", "since ").replace("The ", "the ").replace("  .", " .").strip()


def extract_entities_and_relation(line):
    # make sure line is formatted such that everything is lower cased, except first names
    line = regularize(line).replace(' .', '').strip()

    # extract relation from line
    rel = [w for w in line.split() if w in rel_to_phrases]
    if len(rel) < 1:
        print(f"no relation in line '{line}'")
        return None, None, None
    elif len(rel) > 1:
        print(f"more than one relation in line '{line}'")
        return None, None, None
    else:
        rel = rel[0]

    # print(f"relation {rel} found in line {line}")

    # extract who is e_1 and who is e_2
    e1, e2 = None, None
    first_names = list(set([w for w in line.split() if w[:4] == 'ent_']))
    if len(first_names) != 2:
        print(f"line '{line}' does not have exactly 2 names: {first_names}")
        return rel, None, None
    for p in rel_to_phrases[rel]:
        p = regularize(p)
        # either ( e_1=fn[0] and e_2=fn[1] ) or ( e_1=fn[1] and e_2=fn[0] )
        p1 = p.replace("e_1", first_names[0]).replace("e_2", first_names[1])
        p2 = p.replace("e_1", first_names[1]).replace("e_2", first_names[0])
        # print(f"[line] '{line}'")
        # print(f"[p1]   '{p1}'")
        # print(f"[p2]   '{p2}'")
        # print("")
        if line == p1:
            e1 = first_names[0]
            e2 = first_names[1]
            break
        elif line == p2:
            e1 = first_names[1]
            e2 = first_names[0]
            break

    if None in (e1, e2):
        print(f"could not find entities in '{line}'")
        for p in rel_to_phrases[rel]:
            p = regularize(p)
            p1 = p.replace("e_1", first_names[0]).replace("e_2", first_names[1])
            p2 = p.replace("e_1", first_names[1]).replace("e_2", first_names[0])
            print(f"[p1]   '{p1}'")
            print(f"[p2]   '{p2}'")
            print("")
        # return rel, None, None
    return rel, e1, e2


def valid_proof(story, pl):
    """
    checks if the answer is consistent with the proof and the proof is consistent with the story!
    :param story: the original clean story
    :param pl: predicted line
    :return: proof -is-consistent-with(story) and proof type
    """
    # process story
    story = story.replace("<STORY>", "").strip()

    # get proof
    try:
        proof = regularize(pl.split('<PROOF>')[1].split('<ANSWER>')[0])
    except IndexError:
        print(f"Index error when extracting the proof from '{pl}'")
        return False

    # ignore cases when the model wasn't trained to produce a proof
    if proof == 'none':
        return True

    if '. . .' in proof:
        print(f"model couldn't generate the full proof :(")
        print(proof)
        return False

    # 1 # check if proof is consistent with story

    # (1.1) extract known facts from the story
    facts = []
    for line in story.split('.'):
        if len(line.strip()) <= 0: continue  # skip empty lines
        rel, e1, e2 = extract_entities_and_relation(line)
        if None in (rel, e1, e2): continue  # skip weird story lines
        facts.append((e1, rel, e2))
        # also add the reverse relations
        for inv_rel in inverse_rel[rel]:
            facts.append((e2, inv_rel, e1))
    facts = set(facts)
    # print(f"story: {story}")
    # print(f"facts: {facts}")

    # (1.2) go through the proof steps and extract new facts as we go...
    if args.proof_type == 'reversed':
        proof_steps = proof.split('.')[::-1]
    elif args.proof_type == 'normal':
        proof_steps = proof.split('.')
    else:
        raise ValueError(f"Unknown argument --proof_type {args.proof_type}")

    for statement in proof_steps:
        if len(statement.strip()) <= 0: continue
        # ex: since James is a brother of Rudolf , and Rudolf is the grandson of Jaime, then James is a grandson of Jaime.
        # print(f" statement: {statement}")
        try:
            clause_1 = statement.split("since ")[1].split(" and ")[0].replace(',', '')
            clause_2 = statement.split(" and ")[1].split(" then ")[0].replace(',', '')
        except IndexError:
            print(f"Index error when extracting clauses from '{statement}'")
            return False

        # make sure each clause is actually true and extract their meaning
        rel, e1, e2 = extract_entities_and_relation(clause_1)
        if None in (rel, e1, e2): return False
        if (e1, rel, e2) not in facts:
            print(f"used fact ({e1}-{rel}-{e2}) is not known yet. {facts}")
            return False
        clause_1 = (e1, rel, e2)

        rel, e1, e2 = extract_entities_and_relation(clause_2)
        if None in (rel, e1, e2):
            return False
        if (e1, rel, e2) not in facts:
            print(f"used fact ({e1}-{rel}-{e2}) is not known yet. {facts}")
            return False
        clause_2 = (e1, rel, e2)

        try:
            conclusion = statement.split(" then ")[1].replace(',', '')
        except IndexError:
            print(f"Index error when extracting conclusion from '{statement}'")
            return False
        rel, e1, e2 = extract_entities_and_relation(conclusion)
        if None in (rel, e1, e2):
            return False

        all_entities = (clause_1[0], clause_1[2], clause_2[0], clause_2[2])

        # make sure clause_1 + clause_2 = conclusion
        # clauses and conclusion should be of the form (Y, rel1, X) + (Z, rel2, Y) = (Z, rel3, X)
        if clause_1[0] == clause_2[2] and clause_1[2] == e2 and clause_2[0] == e1:
            # check that rel1 + rel2 == rel3
            rel2 = grel_to_rel[clause_2[1]]
            rel1 = grel_to_rel[clause_1[1]]
            try:
                new_rel = rules['compositional']['family'][rel2][rel1]
            except KeyError:
                print(f"couldn't find {rel2} + {rel1} :(")
                continue
            if new_rel == grel_to_rel[rel]:
                facts.add((e1, rel, e2))
                # also add the reverse relations
                for inv_rel in inverse_rel[rel]:
                    facts.add((e2, inv_rel, e1))
            else:
                print(f"conclusion ({grel_to_rel[rel]}) != new rel ({rel2}+{rel1}={new_rel})")
        elif e1 in all_entities and e2 in all_entities:
            print(f"inconsistent statement `{statement}` will be considered invalid... :(")
            print(f"clause_1[0] '{clause_1[0]}' ?= clause_2[2] '{clause_2[2]}'")
            print(f"clause_1[2] '{clause_1[2]}' ?= e2 '{e2}'")
            print(f"clause_2[0] '{clause_2[0]}' ?= e1 '{e1}'")
        else:
            print(f"incoherent statement `{statement}`")

    # NOTE #
    # Do not check if answer is consistent with proof. Proof can be valid but answer non-valid

    return True


def main():
    print("loading files...")
    with open(args.truth, 'r') as f:
        true_lines = f.readlines()
    with open(args.pred, 'r') as f:
        pred_lines = f.readlines()
    # take only non-empty lines
    pred_lines = list(filter(lambda line: len(line.strip()) > 0, pred_lines))

    assert len(true_lines) == len(pred_lines), f"file length not equal: true={len(true_lines)} pred={len(pred_lines)}"

    correct = []              # answer has the correct first names and relation
    correct_but_invalid = []  # answer has the correct first names and relation but the proof is not valid
    wrong = []                # answer doesn't have the correct relation
    wrong_but_valid = []      # answer doesn't have the correct relation but the proof is valid!

    for idx in range(len(true_lines)):
        print("")
        print(f"--------------------------")
        tl = true_lines[idx]
        pl = pred_lines[idx]

        # get the ground truth story
        story = tl.split('<QUERY>')[0].strip()

        # same line, same query
        query = tl.split('<PROOF>')[0].split('<QUERY>')[1].strip()
        try:
            q2 = pl.split('<PROOF>')[0].split('<QUERY>')[1].strip()
        except IndexError as e:
            print(f"Index error when extracting query from line '{pl}'")
            print(f"original line: '{tl}'")
            wrong.append(idx)
            continue
        assert query == q2, f"query mismatch! true='{query}' pred='{q2}'"
        del q2

        print(f"predicted line: {pl}")

        # check if valid proof and valid answer once and for all
        is_valid = valid_proof(story, pl)
        if not is_valid:
            print(f"invalid proof: {pl}")

        # try to get answers (the last one)
        ans1 = regularize(tl.split('<ANSWER>')[-1])
        try:
            ans2 = regularize(pl.split('<ANSWER>')[-1])
        except IndexError:
            print(f"Index error when extracting the answer from '{pl}'")
            wrong.append(idx)
            if is_valid:
                wrong_but_valid.append(idx)
            continue

        ###
        # (1)
        # check if exact same answer string
        ####
        if ans1 == ans2:
            print(f"same answer! definitely correct")
            correct.append(idx)
            if not is_valid:
                correct_but_invalid.append(idx)
            continue

        # extract e1, e2 and rel from ans1 and ans2
        t_rel, t_e1, t_e2 = extract_entities_and_relation(ans1)  # target answer
        p_rel, p_e1, p_e2 = extract_entities_and_relation(ans2)  # predicted answer

        ###
        # (2)
        # check if relation match
        ###
        if p_rel is None or (p_rel != t_rel and p_rel not in inverse_rel[t_rel]):
            print(f"relation didn't match! (true:{ans1}-vs-pred:{ans2})")
            wrong.append(idx)
            if is_valid:
                wrong_but_valid.append(idx)
            continue

        ###
        # (3)
        # check if first names match
        ###
        if None in (p_e1, p_e2) or len(set([t_e1, t_e2]) - set([p_e1, p_e2])) > 0:
            print(f"not the same first names! (true:{ans1}-vs-pred:{ans2})")
            wrong.append(idx)
            if is_valid:
                wrong_but_valid.append(idx)
            continue

        ###
        # (4)
        # if we reach this point, the first_names match and the relation is either matching or inversed.
        # we still need to check that if the relation is inversed, the entities are also.
        ###

        # check the generated answer.
        is_correct = False
        for p in rel_to_phrases[t_rel]:
            p = regularize(p).replace("e_1", t_e1).replace("e_2", t_e2) + ' .'
            if ans2 == p:
                is_correct = True
                break
        if is_correct:
            print(f"answer match! true:{ans1} = pred:{ans2}")
            correct.append(idx)
            if not is_valid:
                correct_but_invalid.append(idx)
        else:
            print(f"answer didn't match! true:{ans1}-vs-pred:{ans2}")
            wrong.append(idx)
            if is_valid:
                wrong_but_valid.append(idx)

    print("")
    print(f"correct: {len(correct)} / {len(true_lines)} = {len(correct) / len(true_lines)}")
    print(f"correct but invalid: {len(correct_but_invalid)} / {len(true_lines)}"
          f" = {len(correct_but_invalid) / len(true_lines)}")
    print(f"wrong: {len(wrong)} / {len(true_lines)} = {len(wrong) / len(true_lines)}")
    print(f"wrong but valid: {len(wrong_but_valid)} / {len(true_lines)}"
          f" = {len(wrong_but_valid) / len(true_lines)}")

    # save evaluation
    with open(args.pred.replace('.txt', '_eval.yaml'), 'w') as f:
        yaml.safe_dump({
            'correct': {'score': len(correct) / len(true_lines), 'idx': correct},
            'correct_but_invalid': {'score': len(correct_but_invalid) / len(true_lines), 'idx': correct_but_invalid},
            'wrong': {'score': len(wrong) / len(true_lines), 'idx': wrong},
            'wrong_but_valid': {'score': len(wrong_but_valid) / len(true_lines), 'idx': wrong_but_valid}
        }, stream=f)


if __name__ == '__main__':
    args = parser.parse_args()

    with open("../data/relations_store.yaml", 'r') as f:
        relations = yaml.safe_load(f)
    rel_to_phrases, grel_to_rel = get_relation_mappings()
    del relations
    print("done.")
    print("relations -to- phrases:")
    for rel, phrases in rel_to_phrases.items():
        print(f"  {rel}:")
        for p in phrases: print(f"    {p}")

    with open("../data/rules_store.yaml", 'r') as f:
        rules = yaml.safe_load(f)

    inverse_rel = {
        'son': ['father', 'mother'],
        'daughter': ['father', 'mother'],
        'father': ['son', 'daughter'],
        'mother': ['son', 'daughter'],
        'husband': ['wife'],
        'wife': ['husband'],
        'brother': ['brother', 'sister'],
        'sister': ['brother', 'sister'],
        'grandson': ['grandfather', 'grandmother'],
        'granddaughter': ['grandfather', 'grandmother'],
        'grandfather': ['grandson', 'granddaughter'],
        'grandmother': ['grandson', 'granddaughter'],
        'son-in-law': ['father-in-law', 'mother-in-law'],
        'daughter-in-law': ['father-in-law', 'mother-in-law'],
        'father-in-law': ['son-in-law', 'daughter-in-law'],
        'mother-in-law': ['son-in-law', 'daughter-in-law'],
        'brother-in-law': ['brother-in-law', 'sister-in-law'],
        'sister-in-law': ['brother-in-law', 'sister-in-law'],
        'nephew': ['uncle', 'aunt'],
        'niece': ['uncle', 'aunt'],
        'uncle': ['nephew', 'niece'],
        'aunt': ['nephew', 'niece'],
    }

    main()
