import os
import argparse
import json
import yaml
import pandas as pd
import random
import itertools
import pickle as pkl

from sacremoses import MosesTokenizer, MosesDetokenizer

tokenizer = MosesTokenizer(lang='en')
detokenizer = MosesDetokenizer(lang='en')

'''
ex:

detokenizer = MosesDetokenizer(lang='en')

all_test_lines = map(lambda line: tokenizer.tokenize(line.strip(), return_str=True, escape=False), all_test_lines)
all_test_lines = list(map(lambda line: line.replace('< CONTEXT >', '<CONTEXT>')
                          .replace('< QUERY >', '<QUERY>')
                          .replace('< PROOF >', '<PROOF>')
                          .replace('< ANSWER >', '<ANSWER>'), all_test_lines))

test_lines = map(lambda line: detokenizer.detokenize(line.split()), test_lines)
'''

DEBUG = False

MAX_N_HOPS = 100  # to avoid infinite loop when building proof trace
DONE_HOP = 2*MAX_N_HOPS  # constant to break out of the loop and identify that we didn't wait for the max_n_hops!


def my_bool(s):
    if s.strip().lower() in ['yes', '1', 'true']:
        return True
    else:
        return False


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_in', required=True, help="file path to read to generate output text file")
    parser.add_argument('--out_prefix', required=True, help="file path to write the output text file")
    parser.add_argument('--query_only', type=my_bool, default='false',
                        help="generate only the query files (story + question) to let a model generate the rest")
    parser.add_argument('--proof_type', choices=['none', 'short', 'long'], default='none',
                        help="the type of proof to generate: none, direct, or full trace")
    parser.add_argument('--gender_aware', type=my_bool, default='yes',
                        help="replace names by their name+gender")
    parser.add_argument('--reversed', type=my_bool, default='no',
                        help="put the proof AFTER the answer")
    arguments = parser.parse_args()
    return arguments


def print_args():
    print("IN: %s" % args.csv_in)
    print("OUT: %s_[...].txt" % args.out_prefix)
    print("GENDER:", end='\t')
    if args.gender_aware:
        print("- YES (for big train and valid files, replace names by names+gender and use that in split_by_relation_length script")
    else:
        print("- NO (for testing, no need to add the gender to each name)")
    print("QUERY ONLY:", end='\t')
    if args.query_only:
        print("- YES (for testing, only generating prefixes ex: \"<STORY> [...] <QUERY> [...] <PROOF> \")")
    else:
        print("- NO (for training, generating full information)")
    print("REVERSED:", end='\t')
    if args.reversed:
        print("- YES (put the proof AFTER the answer)")
    else:
        print("- NO (put the proof BEFORE the answer)")
    print(f"PROOF TYPE: {args.proof_type}")


def get_all_first_names(df):
    """
    :param df: dataframe
    :return: mapping from gender to list of ( names and binary flag to know if already used or not )
    ex:
     {'male': {name: T/F, name: T/F, ...},
     'female': {name: T/F, name: T/F, ...}}
    """
    fns = {'male': {}, 'female': {}}
    for idx, row in df.iterrows():
        # Get the gender of all characters in the story
        # row['genders'] example: "Kenneth:male,Amy:female,Gail:female"
        genders_list = row['genders'].split(',')
        for person in genders_list:
            name = person.split(':')[0]
            sex = person.split(':')[1]
            fns[sex][name] = False
    return fns


def gender_aware(line, genders):
    """
    :param line: string line
    :param genders: map from "name" to "name[gender]"
    """
    for old_name, new_name in genders.items():
        line = line.replace(old_name, new_name)
    return line


def get_relation_phrases(relations):
    """
    :param relations: yaml dict of relation to partial phrases
    :return: a mapping from relation to all possible phrases
    """
    rel_to_phrases = {}
    for key, val in relations.items():
        for gender in ['male', 'female']:
            rel = val[gender]['rel']
            rel_to_phrases[(key, rel)] = val[gender]['p']

    return rel_to_phrases


def extract_entities_and_relation(line):
    """
    :param line: a factual sentence linking two entities
    :return: relation (key, rel); entity_1; entity_2
    """
    line = line.strip()  # remove trailing spaces
    line = line.replace('The ', 'the ')  # remove caps except for first names

    # extract relation from line
    rel = [w for w in line.split() if w in map(lambda key: key[1], relation_phrases)]
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
    first_names = list(set([w.replace("'s", '') for w in line.split() if w[0].isupper()]))
    if len(first_names) != 2:
        print(f"line '{line}' does not have exactly 2 names: {first_names}")
        return rel, None, None
    for (key, a_rel), phrases in relation_phrases.items():
        if a_rel == rel:
            for p in phrases:
                # either ( e_1=fn[0] and e_2=fn[1] ) or ( e_1=fn[1] and e_2=fn[0] )
                p1 = p.replace("e_1", first_names[0]).replace("e_2", first_names[1]).replace('The ', 'the ')
                p2 = p.replace("e_1", first_names[1]).replace("e_2", first_names[0]).replace('The ', 'the ')
                # print(f"[line] '{line}'")
                # print(f"[p1]   '{p1}'")
                # print(f"[p2]   '{p2}'")
                # print("")
                if line == p1:
                    rel = (key, rel)
                    e1 = first_names[0]
                    e2 = first_names[1]
                    break
                elif line == p2:
                    rel = (key, rel)
                    e1 = first_names[1]
                    e2 = first_names[0]
                    break

    if None in (e1, e2):
        print(f"could not find entities in '{line}'")
        return rel, None, None

    return rel, e1, e2


def triple_to_sent(e1, rel, e2, genders):
    """
    Convert a entity-relation-entity triple to a valid sentence
    :param e1: entity 1
    :param rel: relation from e1 to e2
    :param e2: entity 2
    :param genders: map from "name" to "name[gender]"
    :return: factual sentences linking E1 and E2
    """
    # (1) get gender of e1
    if "[female]" in e1:
        gender = "female"
    elif "[male]" in e1:
        gender = "male"
    else:
        gender = genders[e1].split('[')[1].replace(']', '')

    # (2) get sentence
    sent = relations_store[rel[0]][gender]['p'][0]  # take the 1st sentence arbitrarily
    return sent.replace('e_2', e1).replace('e_1', e2)


def get_new_fact(clause1, clause2, genders):
    """
    Checks if there is a rule that combines the two clauses
    :param clause1: tuple of (E1, relation_1, E2)
    :param clause2: tuple of (E2, relation_2, E3)
    :param genders: map from "name" to "name[gender]"
    :return: (new factual sentences linking E1 and E3 , the relation between E1 and E3)
    """
    e11, rel1, e12 = clause1
    e21, rel2, e22 = clause2
    assert e12 == e21 and e11 != e12 != e22

    # (1) check that rel1 + rel2 exists:
    try:
        new_rel = rules_store['compositional']['family'][rel2][rel1]
    except KeyError:
        # print(f"couldn't find {rel2} + {rel1} :(")
        return None, None

    # (2) get gender of e11
    if "[male]" in e11:
        gender = "male"
    elif "[female]" in e11:
        gender = "female"
    else:
        gender = genders[e11].split('[')[1].replace(']', '')

    # (3) get sentence and new relation
    new_fact = relations_store[new_rel][gender]['p'][0]  # take the 1st sentence arbitrarily
    new_fact = new_fact.replace('e_2', e11).replace('e_1', e22)
    new_rel = (new_rel, relations_store[new_rel][gender]['rel'])

    return new_fact, new_rel


def reverse_triple(e1, rel, e2, genders):
    """
    :param e1: entity 1
    :param rel: relation linking entity 1 to entity 2
    :param e2: entity 2
    :param genders: map from "name" to "name[gender]"
    :return: (e2, inverse_rel, e1)
    """
    # (1) fetch inv_rel
    if rel[0] in rules_store['symmetric']['family']:
        inv_rel = rules_store['symmetric']['family'][rel[0]]
    elif 'inv-' in rel[0]:
        inv_rel = rel[0].replace('inv-', '')
    else:
        inv_rel = f"inv-{rel[0]}"
    # make sure it exists
    if inv_rel not in relations_store:
        raise ValueError(f"{inv_rel} is unknown")

    # (2) fetch gender of e2
    if "[male]" in e2:
        gender = "male"
    elif "[female]" in e2:
        gender = "female"
    else:
        gender = genders[e2].split('[')[1].replace(']', '')

    # (3) replace relation
    inv_rel = (inv_rel, relations_store[inv_rel][gender]['rel'])

    return e2, inv_rel, e1


def build_proof_trace(story_facts, query, proof_txt, genders):
    """
    Build a search path in the space of proofs
    :param story_facts: family graph in string format.
    :param query: the question asked, in Natural Language.
    :param proof_txt: the golden proof
    :param genders: map from "name" to "name[gender]"
    :return: the search to reach the golden proof
    """
    self_debug = False

    ###
    # (1) extract the source and target entities from the query
    ###
    query = query[0].lower() + query[1:]  # lowercase first word
    first_names = [w.replace(",", '') for w in query.split() if w[0].isupper()]
    assert len(first_names) == 2, f"not exactly 2 names in query '{query}'"

    e_1, e_2 = None, None
    for q_phrase in question_store['relational']:
        q_phrase = q_phrase.replace(q_phrase[0], q_phrase[0].lower())  # lowercase first word
        begin = q_phrase.split('e_1')[0]
        if query.startswith(begin):
            e_1 = first_names[0]
            e_2 = first_names[1]
            break
        begin = q_phrase.split('e_2')[0]
        if query.startswith(begin):
            e_1 = first_names[1]
            e_2 = first_names[0]
            break
    if self_debug: print()
    if self_debug: print(f"e1:{e_1} and e2:{e_2} from query '{query}'")

    ###
    # (2) extract correct family relations to be proven from the golden proof
    ###
    proof_steps = {}  # map from entities to relations linking them
    target_rel = None
    for statement in proof_txt.split('.'):
        if len(statement.strip()) <= 0: continue
        # ex: Since A is a <rel1> of B , and B is the <rel2> of C, then A is a <rel3> of C
        # print(f"proof statement: {statement}")
        statement = statement.replace("Since ", '').replace(" and ", ' ').replace(" then ", ' ')
        c1, c2, c3 = statement.split(',')
        rel1, B, A = extract_entities_and_relation(c1)
        rel2, C, B = extract_entities_and_relation(c2)
        rel3, C, A = extract_entities_and_relation(c3)
        proof_steps[(A, B, C)] = (rel1, rel2, rel3)
        # extract the target relation to be found
        if A == e_2 and C == e_1:
            target_rel = rel3
    assert target_rel is not None, f"({e_2}, . , {e_1}) not present in proof steps: {proof_steps}"
    _, inv_target_rel, _ = reverse_triple(e_2, target_rel, e_1, genders)
    target_rel = target_rel[1]
    inv_target_rel = inv_target_rel[1]

    if self_debug: print(f"proof steps: {proof_steps}")
    if self_debug: print(f"To be found: {e_2}-{target_rel}-{e_1}")

    ###
    # (3) Implement a BFS style search on the family tree.
    # the story is made of n sentences: A---B . B---C . C---D . D---E
    # (3.1) order the story to be in a straight line like ^here^
    # (3.2) try to combine 1st hop:          A--C    B--D     C--E
    # (3.3) 2nd hop:                          A--D  A--D B--E  B--E
    # keep both                                A---D      B---E
    # (3.4) last hop:                           A---E    A---E
    #
    # = "a rel b + b rel c = a rel c . b rel c + c rel d = b rel d . c rel d + d rel e = c rel e . <hop>
    #    a rel c + c rel d = a rel d . a rel b + b rel d = a rel d . b rel d + d rel e = b rel e . b rel c + c rel e = b rel d . <hop>
    #    a rel d + d rel e = a rel e . a rel b + b rel d = a rel d.
    #
    # if proof reaches A-B+B-C like in golden proof, then A-rel-C must match
    # otherwise backup and delete the most recent between A-B and B-C
    #
    ###

    # Start by adding the set of known factual sentences and their reverse
    sentences = list(filter(lambda s: len(s) > 0, story_facts.split('.')))  # list of factual sentences to combine
    '''
    sentences_to_add = []  # list of new factual sentences to add
    for sent in sentences:
        rel, B, A = extract_entities_and_relation(sent)
        _, inv_rel, _ = reverse_triple(A, rel, B, genders)
        inv_sent = triple_to_sent(B, inv_rel, A, genders)
        sentences_to_add.append(inv_sent)
    sentences.extend(sentences_to_add)
    '''
    sentences_to_add = []  # list of new factual sentences to add after 1 hop

    proof_trace = ""  # to be built step by step
    done_pairs = set()  # list of sentence pairs already computed : [ ((r1,e12,e11),(r2,r22,r21)) , ... ]
    linked_entities = []  # list of entity-relation-entity tuples
    prefix, hop = "", 0  # indices to know which hop we are at.

    # Then start to go through every possible sentence pair...
    while hop < MAX_N_HOPS:  # max number of hops
        if self_debug: print(f"sentences: {sentences}")
        hop += 1
        # for all possible pair of sentences, try to combine them
        for sent1, sent2 in itertools.combinations(sentences, 2):
            sent1 = extract_entities_and_relation(sent1)  # sent1 = (rel1, e12, e11)
            sent2 = extract_entities_and_relation(sent2)  # sent2 = (rel2, e22, e21)
            # if that pair has already been seen, ignore
            if (sent1, sent2) in done_pairs:
                # print(f"{prefix}already tried {sent1} + {sent2}")
                continue
            # this sentence pair was not seen before

            # Add unordered pair of sentences: A-B and B-A
            done_pairs.add((sent1, sent2))  # done_pairs is a set of ((.,.,.), (.,.,.))
            done_pairs.add((sent2, sent1))
            if self_debug:
                print(f"\n{prefix}new pair: {sent1} + {sent2}")
                print(f"{prefix}done list: {len(done_pairs)}")
                # for e in done_pairs:
                #     print(f"{prefix} {e}")

            rel1, e12, e11 = sent1
            rel2, e22, e21 = sent2
            _, inv_rel1, _ = reverse_triple(e11, rel1, e12, genders)
            _, inv_rel2, _ = reverse_triple(e21, rel2, e22, genders)

            if self_debug: print(f"extracted: {e11}-{rel1}-{e12} and {e21}-{rel2}-{e22}")
            if self_debug: print(f"reversed: {e12}-{inv_rel1}-{e11} and {e22}-{inv_rel2}-{e21}")

            # add linked entities given by the sentences
            if (e11, rel1[1], e12) not in linked_entities:
                linked_entities.append((e11, rel1[1], e12))
                linked_entities.append((e12, inv_rel1[1], e11))
            if (e21, rel2[1], e22) not in linked_entities:
                linked_entities.append((e21, rel2[1], e22))
                linked_entities.append((e22, inv_rel2[1], e21))
            # if (e12, e11) not in linked_entities: linked_entities.append((e12, e11))
            # if (e22, e21) not in linked_entities: linked_entities.append((e22, e21))

            if self_debug:
                print(f"{prefix}linked entities:")
                for ents in linked_entities:
                    print(f"{prefix}  {ents[0]}-{ents[1]}-{ents[2]}")

            # print(f"{prefix}extracted: {e11}-{rel1}-{e12} and {e21}-{rel2}-{e22}")
            # print(f"{prefix}reversed:  {e12}-{inv_rel1}-{e11} and {e22}-{inv_rel2}-{e21}")

            # there are 4 possible ways to combine 2 sentences with 2 entities each (1 in common):
            # either e11 = e21 ; or e11 = e22 ; or e12 = e21 ; or e12 = e22

            if e11 == e21 and e12 != e11 and e12 != e22:
                # in this case, the rel1 must be inverted to link e12--e11e21--e22
                # print("1111111111111111111111111111111111")
                # print(f"({e11}, {rel1}, {e12}) and ({e21}, {rel2}, {e22})")
                # print(f"becomes")
                A, new_rel1, B = e12, inv_rel1, e11
                B, new_rel2, C = e21, rel2, e22
                # also update inv_rel1 now that we updated rel_1
                inv_rel1 = rel1

            elif e11 == e22 and e12 != e11 and e12 != e21:
                # instead of checking inv_rel1 + inv_rel2, reverse the sentences and check rel2 + rel1
                #          (e12, inv-rel1,e) + (e, inv-rel2, e21)
                # check if (e21, rel2, e) + (e, rel1, e12)
                # print("22222222222222222222222222222222")
                # print(f"({e11}, {rel1}, {e12}) and ({e21}, {rel2}, {e22})")
                # print(f"becomes")
                A, new_rel1, B = e21, rel2, e22
                B, new_rel2, C = e11, rel1, e12
                # also swap inv_rel1 and inv_rel2
                tmp = inv_rel1
                inv_rel1 = inv_rel2
                inv_rel2 = tmp

            elif e12 == e21 and e11 != e12 and e11 != e22:
                # check if (e11, rel1, e) + (e, rel2, e22) can make a new link
                # print("3333333333333333333333333333333333333")
                # print(f"({e11}, {rel1}, {e12}) and ({e21}, {rel2}, {e22})")
                # print(f"becomes")
                A, new_rel1, B = e11, rel1, e12
                B, new_rel2, C = e21, rel2, e22

            elif e12 == e22 and e11 != e12 and e11 != e21:
                # in this case, the rel2 must be inverted to link e11--e12e22--e21
                # print("444444444444444444444444444444444444")
                # print(f"({e11}, {rel1}, {e12}) and ({e21}, {rel2}, {e22})")
                # print(f"becomes")
                A, new_rel1, B = e11, rel1, e12
                B, new_rel2, C = e22, inv_rel2, e21
                # also update inv_rel2 now that we updated rel_2
                inv_rel2 = rel2

            else:
                continue

            rel1, rel2 = new_rel1, new_rel2
            if self_debug: print(f"{prefix}trying out ({A}, {rel1}, {B}) + ({B}, {rel2}, {C})")

            # check if (A, rel1, B) + (B, rel2, C) can make a new link
            new_fact, new_rel = get_new_fact((A, rel1[0], B), (B, rel2[0], C), genders)
            if new_fact:
                # print(f"{prefix}linked_entities:{linked_entities}")
                # print(f"{prefix}new triple: {A},{new_rel[1]},{C}")
                if (A, new_rel[1], C) in linked_entities: continue  # ignore if already known
                # (1) add linked entities in both directions: A-->C and C-->A
                linked_entities.append((A, new_rel[1], C))
                _, inv_new_rel, _ = reverse_triple(A, new_rel, C, genders)
                linked_entities.append((C, inv_new_rel[1], A))
                # (2) add factual sentences in both directions: A-->C and C-->A
                sentences_to_add.append(new_fact)
                sentences_to_add.append(triple_to_sent(C, inv_new_rel, A, genders))
                # (3) update proof state
                proof_trace += f"<hop{hop}> since {A} is the {rel1[1]} of {B} and {B} is the {rel2[1]} of {C} then {A} is the {new_rel[1]} of {C}. "
                if self_debug: print(f"{prefix}{A}-{rel1}-{B} + {B}-{rel2}-{C} = {A}-{new_rel}-{C}")
                # stop when we find the answer
                if (A == e_2 and new_rel[1] == target_rel and C == e_1)\
                        or (A == e_1 and new_rel[1] == inv_target_rel and C == e_2):
                    hop = DONE_HOP
                    break
            else:
                if self_debug: print(f"{prefix}{A}-{rel1}-{B} + {B}-{rel2}-{C} = ???")

                if self_debug: print(f"{prefix}trying out ({C}, {inv_rel2}, {B}) + ({B}, {inv_rel1}, {A})")
                # check if (C, inv_rel2, B) + (B, inv_rel1, A) can make a new link

                new_fact, new_rel = get_new_fact((C, inv_rel2[0], B), (B, inv_rel1[0], A), genders)
                if new_fact:
                    # print(f"{prefix}linked_entities:{linked_entities}")
                    # print(f"{prefix}new triple: {C},{new_rel[1]},{A}")
                    if (C, new_rel[1], A) in linked_entities: continue  # ignore if already known
                    # (1) add linked entities in both directions: C-->A and A-->C
                    linked_entities.append((C, new_rel[1], A))
                    _, inv_new_rel, _ = reverse_triple(C, new_rel, A, genders)
                    linked_entities.append((A, inv_new_rel[1], C))
                    # (2) add factual sentences in both directions: C-->A and A-->C
                    sentences_to_add.append(new_fact)
                    sentences_to_add.append(triple_to_sent(A, inv_new_rel, C, genders))
                    # (3) update proof state
                    proof_trace += f"<hop{hop}> since {C} is the {inv_rel2[1]} of {B} and {B} is the {inv_rel1[1]} of {A} then {C} is the {new_rel[1]} of {A}. "
                    if self_debug: print(f"{prefix}{C}-{inv_rel2}-{B} + {B}-{inv_rel1}-{A} = {C}-{new_rel}-{A}")
                    # stop when we find the answer
                    if (C == e_2 and new_rel[1] == target_rel and A == e_1) \
                            or (C == e_1 and new_rel[1] == inv_target_rel and A == e_2):
                        hop = DONE_HOP
                        break
                else:
                    if self_debug: print(f"{prefix}{C}-{inv_rel2}-{B} + {B}-{inv_rel1}-{A} = ???")


        # 1 hop is done.
        prefix = ''.join([" "] * 2 * hop)
        # add all the new facts
        sentences.extend(sentences_to_add)
        sentences_to_add = []
        # print(f"{prefix}sentences: {sentences}")
        # print(f"{prefix}done pairs: {len(done_pairs)}")

    # proof trace finishes like that: " then {A} is the {new_rel[1]} of {C}. "
    # make sure we extracted the correct relation
    last_statement = proof_trace.split(" then ")[-1].replace(' is the ', ' ').replace('. ', '').split()
    assert (last_statement[0] == e_2 and last_statement[1] == target_rel and last_statement[3] == e_1)\
           or (last_statement[0] == e_1 and last_statement[1] == inv_target_rel and last_statement[3] == e_2),\
        f"last statement ({last_statement}) does not agree with targeted link ({e_2}-{target_rel}-{e_1})"
    assert hop == DONE_HOP

    return proof_trace.strip()


def sample_relation():
    rel = random.choice([0, 1, 2])
    if rel == 0:
        # relation sentence of the form: "X is a P to Y"
        rel_1 = 'a'
        rel_2 = 'to'
    elif rel == 1:
        # relation sentence of the form: "X is a P of Y"
        rel_1 = 'a'
        rel_2 = 'of'
    else:
        # relation sentence of the form: "X is the P of Y"
        rel_1 = 'the'
        rel_2 = 'of'
    return rel_1, rel_2


def main():
    out_facts = []  # stories with facts
    out_amt = []  # stories with AMT sentences
    out_both = []  # stories with BOTH AMT sentences AND facts

    for idx, row in df.iterrows():
        if DEBUG and idx == 6:
            break
        if DEBUG: print("--------------------")
        if DEBUG: print("row:", row)
        if DEBUG: print("")
        if DEBUG: print("story facts:", row['story_facts'])
        if DEBUG: print("--------------------")
        if not DEBUG and idx % 1000 == 0:
            print("%d / %d" % (idx, len(df)))

        #################################################
        # Get the gender of all characters in the story #
        #################################################

        # row['genders'] example: "Kenneth:male,Amy:female,Gail:female"
        genders_list = row['genders'].split(',')
        genders = {}
        for person in genders_list:
            name = person.split(':')[0]
            sex = person.split(':')[1]
            genders[name] = "%s[%s]" % (name, sex)

        ########################################
        # factual story # amt story # question #
        ########################################
        story_facts = row['story_facts'].replace('[', '').replace(']', '').replace('  ', ' ').strip()
        story_amt = row['amt_story'].replace('[', '').replace(']', '').replace('  ', ' ').strip()
        query = row['text_query'].replace('[', '').replace(']', '').replace('  ', ' ').strip()

        ##########################
        # total text for queries #
        ##########################
        if args.query_only:
            story_facts = tokenizer.tokenize(story_facts.strip(), return_str=True, escape=False)
            story_amt = tokenizer.tokenize(story_amt.strip(), return_str=True, escape=False)
            query = tokenizer.tokenize(query.strip(), return_str=True, escape=False)
            if args.gender_aware:
                story_facts = gender_aware(story_facts, genders)
                story_amt = gender_aware(story_amt, genders)

            # only write the story and the query, we will let the model generate after the <PROOF> tag
            if args.reversed:
                out_facts.append("<STORY> %s <QUERY> %s <ANSWER> ... <PROOF> ... \n" % (story_facts, query))
                out_amt.append("<STORY> %s <QUERY> %s <ANSWER> ... <PROOF> ... \n" % (story_amt, query))
            else:
                out_facts.append("<STORY> %s <QUERY> %s <PROOF> ... <ANSWER> ... \n" % (story_facts, query))
                out_amt.append("<STORY> %s <QUERY> %s <PROOF> ... <ANSWER> ... \n" % (story_amt, query))
                # out_both.append("<STORY> %s <CLEAN> ... <QUERY> %s <PROOF> ... <ANSWER> ... \n" % (story_amt, query))
            continue

        #########
        # proof #
        #########
        # if not args.query_only:
        if args.proof_type in ('short', 'long'):
            # example of proof_state : [
            #     {('Sara', 'sister', 'Kathryn'): [
            #         ('Sara', 'father', 'John'),
            #         ('John', 'daughter', 'Kathryn')
            #     ]},
            #     {('Sara', 'father', 'John'): [
            #         ('Sara', 'daughter', 'Kristie'),
            #         ('Kristie', 'grandfather', 'John')
            #     ]}
            # ]
            # make proof_state json serializable
            proof_raw = row['proof_state'].replace('), (', ',').replace('(', '').replace(')', '').replace('\', \'', '--').replace('\'', '\"')
            # ex: [
            #     {"Sara-sister-Kathryn": [
            #         "Sara-father-John",
            #         "John-daughter-Kathryn"
            #     ]},
            #     {"Sara-father-John": [
            #         "Sara-daughter-Kristie",
            #         "Kristie-grandfather-John"
            #     ]
            # }]
            proof_raw = json.loads(proof_raw)
            proof_txt = ""
            # loop from the last to the first item since proof_state is built with backward-chaining
            for proof_element in proof_raw[::-1]:
                for result, premises in proof_element.items():
                    if DEBUG: print("result:", result)
                    result = result.split('--')
                    if DEBUG: print("result:", result)
                    if DEBUG: print("premises:", premises)
                    premises[0] = premises[0].split('--')
                    premises[1] = premises[1].split('--')
                    if DEBUG: print("premises:", premises)
                    rel_11, rel_12 = sample_relation()
                    rel_21, rel_22 = sample_relation()
                    rel_31, rel_32 = sample_relation()
                    sent = "Since %s is %s %s %s %s, and %s is %s %s %s %s, then %s is %s %s %s %s. " % (
                        premises[1][2],  # 'John'
                        rel_11,
                        premises[1][1],  # 'grandfather'
                        rel_12,
                        premises[1][0],  # 'Kristie'

                        premises[0][2],  # 'Kristie'
                        rel_21,
                        premises[0][1],  # 'daughter'
                        rel_22,
                        premises[0][0],  # 'Sara'

                        result[2],  # 'John'
                        rel_31,
                        result[1],  # 'father'
                        rel_32,
                        result[0],  # 'Sara'
                    )
                    proof_txt += sent
            proof_txt = proof_txt.strip()

            # if want the proof trace, build it based on the clean story and the golden proof
            if args.proof_type == 'long':
                proof_txt = build_proof_trace(story_facts, query, proof_txt, genders)
        else:
            proof_txt = "none"

        ##########
        # answer #
        ##########
        answer = row['text_target'].replace('[', '').replace(']', '').replace('  ', ' ').strip()

        ####################################
        # tokenization and genderification #
        ####################################
        story_facts = tokenizer.tokenize(story_facts.strip(), return_str=True, escape=False)
        story_amt = tokenizer.tokenize(story_amt.strip(), return_str=True, escape=False)
        query = tokenizer.tokenize(query.strip(), return_str=True, escape=False)
        proof_txt = tokenizer.tokenize(proof_txt.strip(), return_str=True, escape=False)
        proof_txt = proof_txt.replace('< hop1 >', '<hop1>') \
            .replace('< hop2 >', '<hop2>') \
            .replace('< hop3 >', '<hop3>') \
            .replace('< hop4 >', '<hop4>') \
            .replace('< hop5 >', '<hop5>') \
            .replace('< hop6 >', '<hop6>') \
            .replace('< hop7 >', '<hop7>') \
            .replace('< hop8 >', '<hop8>') \
            .replace('< hop9 >', '<hop9>')
        answer = tokenizer.tokenize(answer.strip(), return_str=True, escape=False)

        if args.gender_aware:
            story_facts = gender_aware(story_facts, genders)
            story_amt = gender_aware(story_amt, genders)

        #################
        # save to lists #
        #################
        if args.reversed:
            out_facts.append("<STORY> %s <QUERY> %s <ANSWER> %s <PROOF> %s \n" % (
                story_facts, query, answer, proof_txt
            ))
            out_amt.append("<STORY> %s <QUERY> %s <ANSWER> %s <PROOF> %s \n" % (
                story_amt, query, answer, proof_txt
            ))
        else:
            out_facts.append("<STORY> %s <QUERY> %s <PROOF> %s <ANSWER> %s \n" % (
                story_facts, query, proof_txt, answer
            ))
            out_amt.append("<STORY> %s <QUERY> %s <PROOF> %s <ANSWER> %s \n" % (
                story_amt, query, proof_txt, answer
            ))
            # out_both.append("<STORY> %s <CLEAN> %s <QUERY> %s <PROOF> %s <ANSWER> %s \n" % (
            #     story_amt, story_facts, query, proof_txt, answer
            # ))

        if DEBUG: print(f"out_fact: {out_facts[-1]}")
        if DEBUG: print(f"out_amt: {out_amt[-1]}")
        if DEBUG: print(f"out_both: {out_both[-1]}")
        if DEBUG: print("")

    #################
    # write to file #
    #################
    for text, postfix in zip([out_facts, out_amt, out_both],
                             ['facts', 'amt', 'both']):
        if len(text) > 0:
            with open('%s_%s.txt' % (args.out_prefix, postfix), 'w') as f:
                f.writelines(text)


def extract_first_names():
    # Load first names and save to file if needed
    first_name_file = '/data/1.2345678910_train/first_names.pkl'
    if not os.path.isfile(first_name_file):
        print(f"extracting all first names...")
        first_names = get_all_first_names(df)
        '''
        {'male': {name: T/F, name: T/F, ...},
         'female': {name: T/F, name: T/F, ...}}
        '''
        print(f"got {len(first_names['male'])} males and {len(first_names['female'])} females.")
        print(f"saving to {first_name_file}...")
        with open(first_name_file, 'wb') as f:
            pkl.dump(first_names, f, pkl.HIGHEST_PROTOCOL)
    else:
        print(f"first names already extracted in {first_name_file}")


if __name__ == '__main__':
    args = get_args()
    print_args()

    base_path = os.path.dirname(os.path.realpath(__file__))
    rules_store = yaml.safe_load(open(os.path.join(base_path, 'clutrr/store/rules_store.yaml')))
    question_store = yaml.safe_load(open(os.path.join(base_path, 'clutrr/store/question_store.yaml')))
    relations_store = yaml.safe_load(open(os.path.join(base_path, 'clutrr/store/relations_store.yaml')))

    relation_phrases = get_relation_phrases(relations_store)
    print("Relation phrases:")
    print("relations -to- phrases:")
    for rel, phrases in relation_phrases.items():
        print(f"  {rel}:")
        for p in phrases: print(f"    {p}")

    df = pd.read_csv(args.csv_in)
    extract_first_names()
    main()
