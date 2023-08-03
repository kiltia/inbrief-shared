from clustering.clustering import get_clusters
from searchers.bm25_searcher.bm25_search import BM25Searcher


class Matcher:
    def __init__(self, texts, embeddings, dates):
        self.embeddings = embeddings
        self.texts = texts
        self.dates = dates

    def get_stories(self, match_type="clustering", **args):
        if match_type == "clustering":
            stories_nums = get_clusters(self.embeddings, **args)
        elif match_type == "bm25":
            searcher = BM25Searcher(self.texts, self.embeddings)
            stories_nums = []
            noise_cluster = []
            for i in range(len(self.texts)):
                top, noise = searcher.search_in_deep(text_num=i, **args)
                if top and not (noise):
                    stories_nums.append(top)
                else:
                    noise_cluster.extend(top)
            stories_nums.append(noise_cluster)
        else:
            return None
        stories = []
        for num_texts in stories_nums:
            story_date = [self.dates[i] for i in num_texts]
            num_texts = [x for _, x in sorted(zip(story_date, num_texts))]
            story = []
            for num in num_texts:
                story.append(self.texts[num])
            stories.append(story)
        return stories, stories_nums
