import sys
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance
from text2num import text2num

full_names = ['Atlanta Hawks', 'Boston Celtics', 'Brooklyn Nets', 'Charlotte Hornets',
 'Chicago Bulls', 'Cleveland Cavaliers', 'Detroit Pistons', 'Indiana Pacers',
 'Miami Heat', 'Milwaukee Bucks', 'New York Knicks', 'Orlando Magic',
 'Philadelphia 76ers', 'Toronto Raptors', 'Washington Wizards', 'Dallas Mavericks',
 'Denver Nuggets', 'Golden State Warriors', 'Houston Rockets', 'Los Angeles Clippers',
 'Los Angeles Lakers', 'Memphis Grizzlies', 'Minnesota Timberwolves', 'New Orleans Pelicans',
 'Oklahoma City Thunder', 'Phoenix Suns', 'Portland Trail Blazers', 'Sacramento Kings',
 'San Antonio Spurs', 'Utah Jazz']

cities, teams = set(), set()
ec = {} # equivalence classes
for team in full_names:
    pieces = team.split()
    if len(pieces) == 2:
        ec[team] = [pieces[0], pieces[1]]
        cities.add(pieces[0])
        teams.add(pieces[1])
    elif pieces[0] == "Portland": # only 2-word team
        ec[team] = [pieces[0], " ".join(pieces[1:])]
        cities.add(pieces[0])
        teams.add(" ".join(pieces[1:]))
    else: # must be a 2-word City
        ec[team] = [" ".join(pieces[:2]), pieces[2]]
        cities.add(" ".join(pieces[:2]))
        teams.add(pieces[2])

def same_ent(e1, e2):
    if e1 in cities or e1 in teams or e2 in cities or e2 in teams:
        return e1 == e2 or any((e1 in fullname and e2 in fullname for fullname in full_names))
    else:
        return e1 in e2 or e2 in e1

def int_value(input):
    a_number = False
    try: 
        value = int(input)
        a_number = True
    except ValueError:
        pass

    if not a_number:
        value = text2num(input)
    return value


def trip_match(t1, t2):
    return int_value(t1[1]) == int_value(t2[1]) and t1[2] == t2[2] and same_ent(t1[0], t2[0])

def dedup_triples(triplist):
    """
    this will be inefficient but who cares
    """
    dups = set()
    for i in xrange(1, len(triplist)):
        for j in xrange(i):
            if trip_match(triplist[i], triplist[j]):
                dups.add(i)
                break
    return [thing for i, thing in enumerate(triplist) if i not in dups]

def get_triples(fi):
    all_triples = []
    curr = []
    with open(fi) as f:
        for line in f:
            if line.isspace():
                all_triples.append(dedup_triples(curr))
                curr = []
            else:
                pieces = line.strip().split('|')
                curr.append(tuple(pieces))
    if len(curr) > 0:
        all_triples.append(dedup_triples(curr))
    return all_triples

def calc_precrec(goldfi, predfi):
    gold_triples = get_triples(goldfi)
    pred_triples = get_triples(predfi)
    total_tp, total_predicted, total_gold = 0, 0, 0
    assert len(gold_triples) == len(pred_triples)
    for i, triplist in enumerate(pred_triples):
        tp = sum((1 for j in xrange(len(triplist))
                    if any(trip_match(triplist[j], gold_triples[i][k])
                           for k in xrange(len(gold_triples[i])))))
        total_tp += tp
        total_predicted += len(triplist)
        total_gold += len(gold_triples[i])
    avg_prec = float(total_tp)/total_predicted
    avg_rec = float(total_tp)/total_gold
    print "totals:", total_tp, total_predicted, total_gold
    print "prec:", avg_prec, "rec:", avg_rec
    return avg_prec, avg_rec

def norm_dld(l1, l2):
    ascii_start = 0
    # make a string for l1
    # all triples are unique...
    s1 = ''.join((chr(ascii_start+i) for i in xrange(len(l1))))
    s2 = ''
    next_char = ascii_start + len(s1)
    for j in xrange(len(l2)):
        found = None
        #next_char = chr(ascii_start+len(s1)+j)
        for k in xrange(len(l1)):
            if trip_match(l2[j], l1[k]):
                found = s1[k]
                #next_char = s1[k]
                break
        if found is None:
            s2 += chr(next_char)
            next_char += 1
            assert next_char <= 128
        else:
            s2 += found
    # return 1- , since this thing gives 0 to perfect matches etc
    return 1.0-normalized_damerau_levenshtein_distance(s1, s2)

def calc_dld(goldfi, predfi):
    gold_triples = get_triples(goldfi)
    pred_triples = get_triples(predfi)
    assert len(gold_triples) == len(pred_triples)
    total_score = 0
    for i, triplist in enumerate(pred_triples):
        total_score += norm_dld(triplist, gold_triples[i])
    avg_score = float(total_score)/len(pred_triples)
    print "avg score:", avg_score
    return avg_score

calc_precrec(sys.argv[1], sys.argv[2])
calc_dld(sys.argv[1], sys.argv[2])

# usage python non_rg_metrics.py gold_tuple_fi pred_tuple_fi
