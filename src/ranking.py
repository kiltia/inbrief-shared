import json
import logging
from functools import reduce

from shared.entities import Source

logger = logging.getLogger("app")


class AbstractScorer:
    def get_metrics(self, scores):
        return list(map(self.key, scores))

    def change_scores(self, scores):
        metrics = self.get_metrics(scores)
        max_score = max(metrics)

        return list(
            map(
                lambda pair: (pair[0] * pair[1][0] / max_score, pair[1][1]),
                zip(metrics, scores, strict=True),
            )
        )


class SizeScorer(AbstractScorer):
    def __init__(self):
        self.key = lambda x: len(x[1])


class ReactionScorer(AbstractScorer):
    def __init__(self):
        self.key = lambda x: self._get_story_score(x[1])

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


class CommentScorer(AbstractScorer):
    def __init__(self):
        self.key = lambda x: self._get_story_score(x[1])

    def _get_story_score(self, story_entry):
        return reduce(lambda acc, y: acc + len(y[1].comments), story_entry, 0)


class ViewScorer(AbstractScorer):
    def __init__(self):
        self.key = lambda x: self._get_story_score(x[1])

    def _get_story_score(self, story_entry):
        return reduce(lambda acc, y: acc + y[1].views, story_entry, 0)


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
