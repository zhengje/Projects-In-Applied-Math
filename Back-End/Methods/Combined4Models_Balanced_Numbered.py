"""Four-model controller. Keep beside the two CSVs and four method modules.
Run: python Combined4Models_Balanced.py
Demo: python Combined4Models_Balanced.py --demo
Optional held-out predictions: python Combined4Models_Balanced.py --test-cases test_cases.json
GNN adapter contract is documented in run_gnn. No GNN is fabricated.
"""
import argparse
import importlib
import json
import os
import sys
import pickle
from pathlib import Path
BASE_DIR = str(Path(__file__).resolve().parent)

from collections import Counter
from itertools import combinations
from math import log2
from statistics import mean
from random import Random


# =============================================================================
# 1. JACCARD AND HYBRID FUNCTIONS
# =============================================================================
def jaccard(a, b):
    # Calculate the number of shared games relative to all distinct games.
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a or b else 0.0


def rbo(a, b, p=0.9):
    # Extrapolated rank-biased overlap at equal depth.
    # Compare the rankings at each depth, giving more weight to the top games.
    n = min(len(a), len(b))
    x, y = set(), set()
    score = 0.0
    for d in range(1, n + 1):
        x.add(a[d-1]); y.add(b[d-1])
        overlap = len(x & y) / d
        score += (1-p) * p**(d-1) * overlap
    return score + p**n * overlap


def prepare(results, candidate_limit, top_k):
    if len(results) < 2 or 'Hybrid' in results:
        raise ValueError('Supply at least two individual models, without Hybrid.')
    if not 1 <= top_k <= candidate_limit:
        raise ValueError('Require 1 <= top_k <= candidate_limit.')
    # Remove duplicate recommendations while keeping their original ranking.
    lists = {}
    for name, games in results.items():
        if isinstance(games, (str, bytes)):
            raise TypeError('Supply ranked sequences of IDs.')
        lists[name] = list(dict.fromkeys(games))
    # Use the same number of candidates for each model for a fair comparison.
    depth = min(candidate_limit, *(len(x) for x in lists.values()))
    if depth < top_k:
        raise ValueError('Every model needs at least top_k distinct games.')
    return {name: x[:depth] for name, x in lists.items()}, depth


JACCARD_ALPHA = 0.20

def fuse(lists, top_k=10, rrf_k=60, alpha=JACCARD_ALPHA):
    if rrf_k < 0:
        raise ValueError('rrf_k must be nonnegative.')
    # Start with equal weights and make a small adjustment for agreement.
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must be between 0 and 1.')
    n = len(lists)
    agreement = {a: mean(jaccard(x, y) for b, y in lists.items() if a != b)
                 if n > 1 else 0.0 for a, x in lists.items()}
    total = sum(agreement.values())
    weights = {a: (1-alpha)/n + alpha*v/total if total else 1/n
               for a, v in agreement.items()}
    # Combine model weights with the rank of each recommended game.
    scores = Counter()
    for a,x in lists.items():
        for rank,g in enumerate(x,1):
            scores[g] += weights[a]/(rrf_k+rank)
    # Select the games with the highest combined scores.
    hybrid = sorted(scores, key=lambda g: (-scores[g], str(g)))[:top_k]
    return hybrid, weights, scores


def table(title, headers, rows, note):
    print('\n' + title)
    print('-'*len(title))
    # Format the numerical results into a readable table.
    rows = [[f'{v:.4f}' if isinstance(v,float) else str(v) for v in row] for row in rows]
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i,h in enumerate(headers)]
    for row in [headers]+rows:
        print('  '.join(v.ljust(w) for v,w in zip(row,widths)))
    print(note)


def analyze(results, candidate_limit=100, top_k=10, rrf_k=60, display=None):
    # Prepare the four lists and generate the combined recommendations.
    lists, depth = prepare(results,candidate_limit,top_k)
    hybrid, weights, scores = fuse(lists,top_k,rrf_k)
    # Count how many models recommend each game.
    freq = Counter(g for x in lists.values() for g in x)
    # Print all individual rankings before the combined ranking and analysis.
    if display is not None:
        for name, games in lists.items():
            table(f'{name.upper()} - TOP {top_k} RECOMMENDATIONS',
                  ['Rank', 'Game'],
                  [[i, display[g]] for i, g in enumerate(games[:top_k], 1)], '')
        table(f'FINAL HYBRID - TOP {top_k} RECOMMENDATIONS',
              ['Rank', 'Game', 'Fusion score', 'Support'],
              [[i, display[g], f"{scores[g]:.6f}", freq[g]]
               for i, g in enumerate(hybrid, 1)],
              'Fusion score is not a probability.')
        print('\n' + '=' * 70 + '\nJACCARD AND COMPARATIVE ANALYSIS\n' + '=' * 70)
    print(f'Models: {len(lists)} | Candidates/model: {depth} | Distinct games: {len(freq)} | Top K: {top_k}')
    # Compare each possible pair of individual models.
    pairs = [[a,b,jaccard(lists[a],lists[b]),jaccard(lists[a][:top_k],lists[b][:top_k]),rbo(lists[a],lists[b])] for a,b in combinations(lists,2)]
    table('1. MODEL AGREEMENT',['Model A','Model B',f'J@{depth}',f'J@{top_k}','RBO'],pairs,'Jaccard measures set agreement; RBO emphasizes early ranks. Higher means more agreement, not accuracy.')
    # Find the games that are exclusive to each model and display the weights.
    contributions = [[a,sum(freq[g]==1 for g in x),100*weights[a]] for a,x in lists.items()]
    table('2. MODEL CONTRIBUTIONS',['Model','Unique games','Weight %'],contributions,'Unique games measure exclusive candidate contribution. Fusion weights use 80% equal weighting and 20% Jaccard agreement.')
    # Compare the final hybrid list with each individual method.
    comparisons = [[a,jaccard(x[:top_k],hybrid),rbo(x[:top_k],hybrid),100*len(set(x)&set(hybrid))/top_k] for a,x in lists.items()]
    table('3. HYBRID COMPARISON',['Model','Jaccard','RBO','Candidate coverage %'],comparisons,'Coverage measures how many hybrid games appear anywhere in the individual candidate list.')
    print(f'Mean supporting models: candidate pool {mean(freq.values()):.4f}; hybrid {mean(freq[g] for g in hybrid):.4f}')
    return {'hybrid':hybrid,'weights':weights,'scores':scores,'pairwise':pairs,'contributions':contributions,'hybrid_comparison':comparisons}


# =============================================================================
# 2. PREDICTIVE ACCURACY FUNCTIONS
# =============================================================================
def metrics(ranking, relevant, k):
    # Check which recommended games appear in the held-out relevant games.
    hits = [int(g in relevant) for g in ranking[:k]]
    # Calculate ranking accuracy, with more importance given to higher ranks.
    dcg = sum(h/log2(i+2) for i,h in enumerate(hits))
    ideal = sum(1/log2(i+2) for i in range(min(k,len(relevant))))
    return (sum(hits)/k, sum(hits)/len(relevant), dcg/ideal,
            next((1/(i+1) for i,h in enumerate(hits) if h),0.0))


def paired_ci(diffs, rng, repeats=2000):
    if len(diffs)<2:
        return 'N/A', 'N/A'
    # Resample the user-level differences to estimate a 95% confidence interval.
    samples = sorted(mean(rng.choices(diffs,k=len(diffs))) for _ in range(repeats))
    return samples[int(.025*repeats)], samples[int(.975*repeats)-1]


def evaluate(test_cases, candidate_limit=100, top_k=10, rrf_k=60):
    """One case per independent test user; train without held-out interactions.
    All predictions must exclude seen games and use the same eligible catalogue.
    """
    # Store the accuracy results for each model and reduced hybrid.
    values, ablation = {}, {}
    expected = None
    for case in test_cases:
        relevant = set(case['relevant'])
        if not relevant:
            continue
        lists,_ = prepare(case['predictions'],candidate_limit,top_k)
        if expected is None: expected = set(lists)
        if set(lists)!=expected: raise ValueError('Model names must match across users.')
        # Generate the hybrid separately for each test user.
        hybrid,_,_ = fuse(lists,top_k,rrf_k)
        for a,x in {**lists,'Hybrid':hybrid}.items():
            values.setdefault(a,[]).append(metrics(x,relevant,top_k))
        # Remove one model at a time to measure its effect on hybrid accuracy.
        for a in lists:
            reduced,_,_ = fuse({b:x for b,x in lists.items() if b!=a},top_k,rrf_k)
            ablation.setdefault(a,[]).append(metrics(reduced,relevant,top_k)[2])
    if not values: raise ValueError('No users with held-out relevant games.')
    # Average the accuracy metrics across all test users.
    avg = {a:tuple(mean(col) for col in zip(*rows)) for a,rows in values.items()}
    # Rank the methods by their average NDCG score.
    ordered = sorted(avg,key=lambda a:-avg[a][2])
    n = len(values['Hybrid'])
    table(f'11. PREDICTIVE ACCURACY ({n} USERS)',['Model','Precision','Recall','NDCG','MRR'],[[a,*avg[a]] for a in ordered],f'Macro averages at K={top_k}. Ranked by NDCG; higher is better.')
    base = avg['Hybrid'][2]
    rows = [[a,mean(x),base-mean(x)] for a,x in ablation.items()]
    table('12. ACCURACY ABLATION',['Removed model','Reduced NDCG','NDCG loss'],rows,'Positive loss means removal hurts accuracy; negative means removal helps.')
    rng = Random(42)
    # Compare the hybrid against each individual model using paired user results.
    reliability = []
    for a in values:
        if a=='Hybrid': continue
        diffs = [h[2]-v[2] for h,v in zip(values['Hybrid'],values[a])]
        lo,hi = paired_ci(diffs,rng)
        reliability.append([a,mean(diffs),lo,hi,100*mean(d>1e-12 for d in diffs),100*mean(abs(d)<=1e-12 for d in diffs)])
    table('13. HYBRID RELIABILITY',['Compared with','NDCG gain','95% CI low','95% CI high','Win %','Tie %'],reliability,'Paired user bootstrap intervals (unadjusted); positive gain favors hybrid. Win rate excludes ties.')
    print('Best observed NDCG: '+', '.join(a for a in ordered if abs(avg[a][2]-avg[ordered[0]][2])<1e-12))
    return {'averages':avg,'ablation':rows,'reliability':reliability,'ranking':ordered}



# ============================================================
# 3. GNN INTEGRATION - CALL THE EXISTING GNN FILE
# ============================================================
def run_gnn(input_games, subjective_path, objective_path, top_n):
    # Call the existing GNN directly so it returns up to 100 candidates.
    # The GNN loads its own CSV files and saved model from the working folder.
    module = importlib.import_module("FinalGNNMethod")
    return module.get_recommendations(input_games, top_k=top_n)


# =============================================================================
# 4. STANDARDIZE AND DISPLAY RECOMMENDATIONS
# =============================================================================
def normalize_titles(results, excluded=()):
    """Match original controller's title-based identity convention."""
    if isinstance(results, tuple) and len(results) == 2:
        results = results[1]
    if hasattr(results, "columns"):
        results = results.to_dict("records")
    if isinstance(results, (str, bytes, dict)):
        raise TypeError("Expected a ranked list of titles or title dictionaries.")
    # Standardize game titles and exclude the games already entered by the user.
    blocked = {str(g).strip().casefold() for g in excluded}
    titles = {}
    for rec in results:
        title = rec.get("title") if isinstance(rec, dict) else rec
        if not isinstance(title, str) or not title.strip():
            raise ValueError("Every recommendation must have a nonempty title.")
        key = title.strip().casefold()
        if key not in blocked:
            titles.setdefault(key, title.strip())
    return titles


# =============================================================================
# EXTENDED JACCARD ANALYSIS FOR THE REPORT
# =============================================================================
def extended_jaccard(lists, result, display, top_k):
    names = list(lists)
    depth = len(lists[names[0]])
    hybrid = result['hybrid']
    sets = {a: set(x) for a, x in lists.items()}
    freq = Counter(g for x in lists.values() for g in x)
    pairs = list(combinations(names, 2))
    agreement = {a: mean(jaccard(sets[a], sets[b]) for b in names if b != a)
                 for a in names}
    print('\n' + '='*70 + '\nEXTENDED JACCARD REPORT ANALYSIS\n' + '='*70)

    # Compare agreement at several depths rather than just the top 10 and 100.
    depths = sorted({k for k in (5, top_k, 20, 50, depth) if k <= depth})
    table('5. AGREEMENT ACROSS RANKING DEPTHS', ['Model A','Model B','Shared','Union'] + [f'J@{k}' for k in depths],
          [[a,b,len(sets[a]&sets[b]),len(sets[a]|sets[b])] +
           [jaccard(lists[a][:k],lists[b][:k]) for k in depths] for a,b in pairs],
          'Shared and union counts refer to the full candidate lists.')
    strongest = max(pairs, key=lambda p: jaccard(sets[p[0]],sets[p[1]]))
    weakest = min(pairs, key=lambda p: jaccard(sets[p[0]],sets[p[1]]))
    for label,(a,b) in [('Strongest',strongest),('Weakest',weakest)]:
        print(f'{label} pair: {a} / {b}, J={jaccard(sets[a],sets[b]):.4f}, shared games={len(sets[a]&sets[b])}.')

    # Show how much of each list is exclusive and how weights are adjusted.
    old = {a: 1/(1+sum(jaccard(sets[a],sets[b]) for b in names if b != a)) for a in names}
    table('6. REDUNDANCY AND WEIGHT ADJUSTMENT',
          ['Model','Mean J','Exclusive %','Old weight %','New weight %','Change vs equal (pp)'],
          [[a,agreement[a],100*sum(freq[g]==1 for g in lists[a])/depth,
            100*old[a]/sum(old.values()),100*result['weights'][a],
            100*(result['weights'][a]-1/len(names))] for a in names],
          'Old weights rewarded disagreement; new weights stay near equal and reward agreement modestly.')
    low = min(names,key=agreement.get)
    print(f'{low} has the lowest mean agreement ({agreement[low]:.4f}); this is not evidence of low accuracy.')

    # Measure whether the hybrid concentrates recommendations shared by models.
    table('7. CONSENSUS DISTRIBUTION', ['Supporting models','Pool count','Pool %','Hybrid count','Hybrid %'],
          [[n,sum(v==n for v in freq.values()),100*sum(v==n for v in freq.values())/len(freq),
            sum(freq[g]==n for g in hybrid),100*sum(freq[g]==n for g in hybrid)/len(hybrid)]
           for n in range(1,len(names)+1)], 'Support counts membership in full candidate lists.')
    pool_mean = mean(freq.values()); hybrid_mean = mean(freq[g] for g in hybrid)
    print(f'Mean support rises from {pool_mean:.4f} to {hybrid_mean:.4f} ({100*(hybrid_mean/pool_mean-1):+.2f}%).')

    # Compare weighting choices using the same candidate lists and RRF constant.
    rows = []
    for alpha in (0.0, 0.1, JACCARD_ALPHA, 0.5, 1.0):
        ranking,w,_ = fuse(lists,top_k,alpha=alpha)
        rows.append([alpha,*[100*w[a] for a in names],jaccard(hybrid,ranking),
                     len(set(hybrid)-set(ranking)),mean(jaccard(ranking,lists[a][:top_k]) for a in names),
                     mean(freq[g] for g in ranking)])
    table('8. WEIGHTING SENSITIVITY', ['Alpha',*[a+' %' for a in names],'J vs final','Replaced','Mean J@top','Mean support'],rows,
          'Alpha=0 is equal weighting; alpha=1 is full agreement weighting. Stability is measured against alpha=0.20.')

    # Identify the actual games gained and lost when each method is removed.
    rows = []
    for a in names:
        reduced,_,_ = fuse({b:x for b,x in lists.items() if b!=a},top_k)
        lost = [g for g in hybrid if g not in reduced]
        gained = [g for g in reduced if g not in hybrid]
        rows.append([a,jaccard(hybrid,reduced),len(lost),100*len(lost)/len(hybrid)])
        print(f'\nWithout {a}: lost: '+(', '.join(display[g] for g in lost) or 'None'))
        print('Added: '+(', '.join(display[g] for g in gained) or 'None'))
    table('9. MODEL REMOVAL STABILITY',['Removed','J vs final','Replaced','Change %'],rows,
          'Removal recalculates weights among remaining models; it measures influence, not accuracy.')
    unique = {a:sum(freq[g]==1 for g in lists[a]) for a in names}
    most_unique = max(unique.values()); most_removed = max(r[2] for r in rows)
    print('\n10. REPORT CONCLUSIONS')
    print('Most exclusive contribution: '+', '.join(a for a in names if unique[a]==most_unique)+f' ({most_unique} games each).')
    print('Greatest removal influence: '+', '.join(r[0] for r in rows if r[2]==most_removed)+f' ({most_removed} replacements each).')
    print(f'Balanced hybrid mean Jaccard against individual top lists: {mean(jaccard(hybrid,lists[a][:top_k]) for a in names):.4f}.')
    print(f'Balanced weights range from {100*min(result["weights"].values()):.2f}% to {100*max(result["weights"].values()):.2f}%.')
    print('These findings describe this input profile only. Use the separate held-out experiment for accuracy conclusions.')


def report(raw, excluded=(), candidate_limit=100, top_k=10):
    # Keep the original titles for printing and standardized titles for comparison.
    display = {}
    # Remove duplicate recommendations while keeping their original ranking.
    lists = {}
    for name, recs in raw.items():
        titles = normalize_titles(recs, excluded)
        lists[name] = list(titles)
        for key, title in titles.items():
            display.setdefault(key, title)
    # Display the selected games before showing the recommendation lists.
    if excluded:
        print("\nSELECTED GAMES: " + ", ".join(excluded))
    # Print individual and hybrid rankings, then run the numerical analysis.
    result = analyze(lists, candidate_limit, top_k, display=display)
    prepared, depth = prepare(lists, candidate_limit, top_k)
    hybrid = result['hybrid']
    # Check how many final recommendations change when each model is removed.
    table('4. MODEL REMOVAL SENSITIVITY', ['Removed','Top-K replaced'],
          [[name, len(set(hybrid)-set(fuse({b:x for b,x in prepared.items() if b!=name},top_k)[0]))]
           for name in prepared], 'Measures ranking influence, not predictive accuracy.')
    extended_jaccard(prepared, result, display, top_k)
    return result


def load_test_cases(path):
    """JSON list of held-out cases, all using titles. Models must be trained
    without held-out interactions before generating these predictions."""
    # Load test predictions generated separately using training-only models.
    with open(path, encoding='utf-8') as f:
        cases = json.load(f)
    output = []
    for case in cases:
        if set(case['predictions']) != {'KNN','Similarity','ALS','GNN'}:
            raise ValueError('Each case must include all four model names.')
        seen = case.get('seen', [])
        relevant = list(normalize_titles(case['relevant'], seen))
        output.append({'predictions': {a:list(normalize_titles(x,seen)) for a,x in case['predictions'].items()}, 'relevant':relevant})
    return output


# =============================================================================
# 5. MAIN PROGRAM - RUN ALL FOUR MODELS
# =============================================================================
def main(test_cases_path=None):
    import numpy as np
    import pandas as pd
    from scipy.sparse import csr_matrix
    # Use the same folder for the datasets and external method files.
    os.chdir(BASE_DIR)
    # Import the existing recommendation methods rather than rewriting them.
    import FinalKNNMethod as knn_script
    import FinalSimilarityMatrixMethod as sim_script
    import FinalIMFMethod as imf_script

    # =============================================================================
    # STEP 1: LOAD DATASETS AND GET USER INPUT
    # =============================================================================
    dataset = pd.read_csv(os.path.join(BASE_DIR, "Data_Objective.csv"))
    print(f"Fixed metadata dataset loaded: {len(dataset)} games")

    # Pass the interaction dataset path to the external methods instead of loading it twice.
    SUBJECTIVE_PATH = os.path.join(BASE_DIR, "Data_Subjective.csv")
    if not os.path.isfile(SUBJECTIVE_PATH):
        raise FileNotFoundError(SUBJECTIVE_PATH)

    # Set the number of candidates from each model and final recommendations.
    CANDIDATES_PER_MODEL = 100
    FINAL_TOP_N = 10
    N_RECS = CANDIDATES_PER_MODEL
    print(
        "\nEnter as many games as you want to create a combined "
        "hybrid recommendation profile."
    )
    print("Press ENTER on a blank line when finished.\n")

    # Allow the user to enter multiple games without duplicates.
    input_games = []

    while True:
        title = input(
            f"Enter game {len(input_games) + 1}: "
        ).strip()

        if not title:
            if input_games:
                break
            print("Please enter at least one game.")
            continue

        if title.casefold() in {
            x.casefold() for x in input_games
        }:
            print("Game already entered.")
            continue

        input_games.append(title)

    # =============================================================================
    # STEP 2: GET THE FOUR RECOMMENDATION LISTS
    # =============================================================================
    # MODEL 1: KNN Content Model
    if hasattr(knn_script, "recommend_games_knn"):
        knn_fn = knn_script.recommend_games_knn
        knn_data = getattr(knn_script, "df", dataset)
    elif hasattr(knn_script, "build_knn"):
        knn_fn, knn_data = knn_script.build_knn()
    else:
        raise AttributeError("Could not find 'recommend_games_knn' or 'build_knn' in FinalKNNMethod.py")

    # Call the existing KNN method and collect its ranked recommendations.
    _, knn_results = knn_fn(input_games, knn_data, top_n=N_RECS)
    if not isinstance(knn_results, list):
        raise RuntimeError(f"KNN Error: {knn_results}")

    # MODEL 2: Cosine Similarity Matrix
    if hasattr(sim_script, "recommend_games"):
        sim_fn = sim_script.recommend_games
        sim_data = getattr(sim_script, "df", dataset)
    elif hasattr(sim_script, "build_similarity"):
        sim_fn, sim_data = sim_script.build_similarity()
    else:
        raise AttributeError("Could not find 'recommend_games' or 'build_similarity' in FinalSimilarityMatrixMethod.py")

    # Call the existing similarity method and collect its ranked recommendations.
    _, sim_results = sim_fn(input_games, sim_data, top_n=N_RECS)
    if not isinstance(sim_results, list):
        raise RuntimeError(f"Similarity Error: {sim_results}")

    # MODEL 3: Implicit Matrix Factorization
    # Load the saved ALS model if available, otherwise train it using the existing method.
    pkl_path = os.path.join(BASE_DIR, "implicit_mf_model.pkl")
    if os.path.exists(pkl_path):
        print(f"Loading pre-trained ALS model from {pkl_path}...")
        with open(pkl_path, "rb") as f:
            artifacts = pickle.load(f)
        model = artifacts["model"]
        app_index_map = artifacts["app_index_map"]
        app_id_to_index_map = artifacts["app_id_to_index_map"]
        game_title = artifacts["game_title"]
        title_to_app_id_map = artifacts["title_to_app_id_map"]
    else:
        print("Training Implicit ALS model via FinalIMFMethod...")
        model, _, _, app_index_map, app_id_to_index_map, game_title, title_to_app_id_map = imf_script.implicit_mf()
        with open(pkl_path, "wb") as f:
            pickle.dump({
                "model": model,
                "app_index_map": app_index_map,
                "app_id_to_index_map": app_id_to_index_map,
                "game_title": game_title,
                "title_to_app_id_map": title_to_app_id_map,
            }, f)
        print(f"Saved trained ALS model to {pkl_path}")

    # Match the games entered by the user to the ALS game IDs.
    valid_app_ids = []
    for name in input_games:
        search_name = name.strip().lower()
        app_id = title_to_app_id_map.get(search_name)

        if app_id is None or app_id not in app_id_to_index_map:
            for cand_title, cand_id in title_to_app_id_map.items():
                if search_name in cand_title and cand_id in app_id_to_index_map:
                    app_id = cand_id
                    break

        if app_id is not None and app_id in app_id_to_index_map and app_id not in valid_app_ids:
            valid_app_ids.append(app_id)

    if not valid_app_ids:
        raise RuntimeError(f"None of {input_games} matched in the Implicit ALS interaction matrix.")

    # Create a new user profile from the selected games.
    id_indices = [app_id_to_index_map[aid] for aid in valid_app_ids]
    user_items = csr_matrix(
        (np.ones(len(id_indices), dtype=np.float32), ([0] * len(id_indices), id_indices)),
        shape=(1, model.item_factors.shape[0])
    )

    try:
        # Use the new profile to generate ALS recommendations.
        ids, scores = model.recommend(
            userid=0,
            user_items=user_items,
            N=N_RECS,
            filter_already_liked_items=True,
            recalculate_user=True
        )
    except TypeError as exc:
        raise RuntimeError("ALS must support recalculate_user=True for a new profile.") from exc

    if isinstance(ids, np.ndarray) and ids.ndim > 1:
        ids = ids[0]

    # Convert the ALS recommendation indices back to game titles.
    implicit_results = []
    for idx in ids:
        if int(idx) < 0 or int(idx) >= model.item_factors.shape[0]:
            continue
        rec_app_id = app_index_map[int(idx)]
        rec_title = game_title.get(rec_app_id)
        if rec_title:
            implicit_results.append({"title": rec_title})

    print("\nSuccessfully gathered list arrays:")
    print(f" -> KNN Recs Count: {len(knn_results)}")
    print(f" -> Similarity Recs Count: {len(sim_results)}")
    print(f" -> Implicit ALS Recs Count: {len(implicit_results)}")


    # MODEL 4: Call the external GNN method using the adapter above.
    gnn_results = run_gnn(input_games, SUBJECTIVE_PATH,
                          os.path.join(BASE_DIR, "Data_Objective.csv"), N_RECS)
    print(f" -> GNN Recs Count: {len(gnn_results)}") if isinstance(gnn_results, list) else None
    # -------------------------------------------------------------------------
    # STEP 3: COMBINE THE MODELS AND RUN THE ANALYSIS
    # -------------------------------------------------------------------------
    # Collect all four recommendation lists in one dictionary.
    raw = {"KNN": knn_results, "Similarity": sim_results,
           "ALS": implicit_results, "GNN": gnn_results}
    # Run the full Jaccard analysis and print the combined recommendations.
    result = report(raw, input_games, N_RECS, FINAL_TOP_N)
    # -------------------------------------------------------------------------
    # STEP 4: EVALUATE PREDICTIVE ACCURACY (IF TEST DATA IS AVAILABLE)
    # -------------------------------------------------------------------------
    # Only calculate predictive accuracy if held-out test predictions are provided.
    if test_cases_path:
        evaluate(load_test_cases(test_cases_path), N_RECS, FINAL_TOP_N)
    else:
        print("\nPredictive accuracy not evaluated: supply leakage-free held-out predictions with --test-cases.")
    return result

# =============================================================================
# 6. EXAMPLE ANALYSIS USING SYNTHETIC DATA
# =============================================================================
def demo():
    # Synthetic rankings with noisy shared and model-specific preferences.
    # Use a fixed seed so the example results can be reproduced.
    rng = Random(73)
    cases = []
    for _ in range(200):
        catalogue = list(range(300))
        relevant = set(rng.sample(catalogue,15))
        shared = {g:rng.gauss(0,1) for g in catalogue}
        predictions = {}
        for name,signal in [('KNN',1.0),('Similarity',0.9),('ALS',1.4),('GNN',1.6)]:
            score = {g:0.65*shared[g]+rng.gauss(0,1)+signal*(g in relevant) for g in catalogue}
            predictions[name] = sorted(catalogue,key=lambda g:-score[g])[:100]
        cases.append({'predictions':predictions,'relevant':relevant})
    print('SYNTHETIC DEMONSTRATION — NOT STEAM DATA OR TRAINED MODELS')
    report({a:[f"Game {g}" for g in x] for a,x in cases[0]['predictions'].items()})
    evaluate(cases)


# =============================================================================
# 7. START PROGRAM
# =============================================================================
if __name__ == '__main__':
    # Choose between the actual recommender and the synthetic example.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--test-cases', type=lambda p: str(Path(p).resolve()))
    args = parser.parse_args()
    if args.demo:
        demo()
    else:
        main(args.test_cases)
