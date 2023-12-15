class Scorer:
    pass


class SizeScorer(Scorer):
    def change_scores(self, scores):
        sizes = list(map(lambda x: len(x[1]), scores))
        max_size = max(sizes)

        return map(
            lambda pair: (pair[0] * pair[1][0] / max_size, pair[1][1]),
            zip(sizes, scores, strict=True),
        )


class Ranker:
    def __init__(self, scorers):
        self.scorers = list(map(lambda x: x(), scorers))

    def get_sorted(self, stories, return_scores=False):
        current_scores = list(map(lambda x: (1.0, x), stories))
        for scorer in self.scorers:
            current_scores = scorer.change_scores(current_scores)

        sorted_scores = sorted(current_scores, key=lambda x: x[0])

        if return_scores:
            sorted_scores = map(lambda x: x[1], sorted_scores)

        return list(sorted_scores)
