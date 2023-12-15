import json
import logging
from functools import reduce

from shared.entities import Source

logger = logging.getLogger("app")


class Scorer:
    pass


class SizeScorer(Scorer):
    def change_scores(self, scores):
        sizes = list(map(lambda x: len(x[1]), scores))
        max_size = max(sizes)

        return list(
            map(
                lambda pair: (pair[0] * pair[1][0] / max_size, pair[1][1]),
                zip(sizes, scores, strict=True),
            )
        )


class ReactionScorer(Scorer):
    def _get_reactions(self, entity: Source):
        if entity.reactions is None:
            return 0

        reactions = json.loads(entity.reactions)

        count = 0
        for reaction_entry in reactions:
            count += reaction_entry["count"]

        return count

    def _get_story_score(self, story_entry):
        return reduce(
            lambda acc, y: acc + self._get_reactions(y[1]), story_entry, 0
        )

    def change_scores(self, scores):
        counts = list(map(lambda x: self._get_story_score(x[1]), scores))

        max_count = max(counts)

        return list(
            map(
                lambda pair: (
                    (1 + pair[0]) * pair[1][0] / (1 + max_count),
                    pair[1][1],
                ),
                zip(counts, scores, strict=True),
            )
        )


class Ranker:
    def __init__(self, scorers):
        self.scorers = list(map(lambda x: x(), scorers))

    def get_sorted(self, stories, return_scores=False):
        current_scores = list(map(lambda x: (1.0, x), stories))
        for scorer in self.scorers:
            current_scores = scorer.change_scores(current_scores)

        sorted_scores = sorted(
            current_scores, key=lambda x: x[0], reverse=True
        )

        if return_scores:
            sorted_scores = map(lambda x: x[1], sorted_scores)

        return list(sorted_scores)
