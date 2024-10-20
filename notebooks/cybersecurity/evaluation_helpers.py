from langfuse import Langfuse

langfuse = Langfuse()


def validate_answer(example, pred, trace=None):
    return example.output.label == pred.output.label


def fetch_traces(run_id, limit=100):
    traces = langfuse.fetch_traces(session_id=run_id, limit=limit)
    results = []
    score_client = langfuse.client.score
    for t in traces.data:
        scores = score_client.get(score_ids=",".join(t.scores))
        res = dict()
        for s in scores.data:
            res[s.name] = s.value if s.value is not None else s.string_value
        results.append(res)
    return results


def calculate_metrics(results, classes):
    metrics = dict()
    for c in classes:
        metrics[c] = dict()
        metrics[c]["tp"] = 0
        metrics[c]["fp"] = 0
        metrics[c]["fn"] = 0
        metrics[c]["tn"] = 0
        metrics[c]["precision"] = 0
        metrics[c]["recall"] = 0
        metrics[c]["f1"] = 0
    for r in results:
        for c in classes:
            if r["ground_truth"] == c:
                if r["prediction"] == c:
                    metrics[c]["tp"] += 1
                else:
                    metrics[c]["fn"] += 1
            else:
                if r["prediction"] == c:
                    metrics[c]["fp"] += 1
                else:
                    metrics[c]["tn"] += 1
    for c in classes:
        tp = metrics[c]["tp"]
        fp = metrics[c]["fp"]
        fn = metrics[c]["fn"]
        tn = metrics[c]["tn"]
        metrics[c]["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics[c]["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics[c]["f1"] = 2 * (metrics[c]["precision"] * metrics[c]["recall"]) / (
                metrics[c]["precision"] + metrics[c]["recall"]) if (metrics[c]["precision"] + metrics[c][
            "recall"]) > 0 else 0
    metrics["macro"] = dict()
    metrics["macro"]["precision"] = sum([metrics[c]["precision"] for c in classes]) / len(classes)
    metrics["macro"]["recall"] = sum([metrics[c]["recall"] for c in classes]) / len(classes)
    metrics["macro"]["f1"] = sum([metrics[c]["f1"] for c in classes]) / len(classes)
    return metrics
