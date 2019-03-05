from gensim.models import KeyedVectors
import logging
import sys


# todo logging levels
# todo docstrings
class BilangTranslator:
    lang1 = None
    lang2 = None
    embeddings_fn1 = None
    embeddings_fn2 = None
    models = {}

    best_lang1_to_lang2_translations = []

    translations = {}

    logger = logging.getLogger("BilangTranslator")
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('\t'.join(["%(asctime)s", "%(name)s", "%(levelname)s", "%(message)s"]))
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    def __init__(self, lang1, lang2, embeddings_fn1, embeddings_fn2, initialize_translations=True):
        self.lang1 = lang1
        self.lang2 = lang2
        self.embeddings_fn1 = embeddings_fn1
        self.embeddings_fn2 = embeddings_fn2
        self.logger.info("BEFORE load vectors fn 1")
        self.models[self.lang1] = KeyedVectors.load_word2vec_format(embeddings_fn1)
        self.logger.info("AFTER load vectors fn 1")
        self.logger.info("BEFORE load vectors fn 2")
        self.models[self.lang2] = KeyedVectors.load_word2vec_format(embeddings_fn2)
        self.logger.info("AFTER load vectors fn 2")

        self.translations[lang1] = {}
        self.translations[lang2] = {}
        if initialize_translations:
            self.initialize_translations()

    def _translate_naive(self, src_lang, tgt_lang, src_word, n=10):
        assert src_lang in (self.lang1, self.lang2)
        assert tgt_lang in (self.lang1, self.lang2)

        # src_model = self.models[src_lang]
        # tgt_model = self.models[tgt_lang]

        src_vec = None

        res = []
        try:
            src_vec = self.get_vec_by_word(src_lang, src_word)
            nearest_translations_words = self.get_nearest_neighbors(tgt_lang, src_vec, n=n)

            for translation_candidate_word, distance in nearest_translations_words:
                res.append((src_word, translation_candidate_word, distance))

        except KeyError as e:
            self.logger.warning(e)

        return res

    def translate_naive(self, src_lang, tgt_lang, src_word):
        return self._translate_naive(src_lang, tgt_lang, src_word)

    # todo implement
    def translate_morfessor_boosted(self, src_lang, tgt_lang, src_word):
        naive_translations = self._translate_naive(src_lang, tgt_lang, src_word)
        raise NotImplementedError()

    def get_vec_by_word(self, lang, word):
        return self.models[lang][word]

    def get_best_translation_candidates(self):
        return self.best_lang1_to_lang2_translations

    def set_best_translation_candidates(self, candidates):
        for (src_lang, src_word), (tgt_lang, tgt_word) in candidates:
            assert src_lang in (self.lang1, self.lang2)
            assert tgt_lang in (self.lang1, self.lang2)
            assert src_lang != tgt_lang

            if src_lang == self.lang1:
                self.best_lang1_to_lang2_translations.append((src_word, tgt_word))
            else:
                # src_lang == self.lang2
                self.best_lang1_to_lang2_translations.append((tgt_word, src_word))

    def set_best_translation_candidates_from_tsv(self, left_lang, right_lang, tsv_fn):
        assert left_lang in (self.lang1, self.lang2)
        assert right_lang in (self.lang1, self.lang2)
        assert left_lang != right_lang

        translation_candidates_list = []
        with open(tsv_fn, 'r', encoding="utf-8") as tsv_f:
            for line in tsv_f:
                line_splitted = line.rstrip().split('\t')
                if len(line_splitted) != 2:
                    self.logger.warning(f"in set_best_translation_candidates_from_tsv(): bad line, skipping: {line}")
                    continue
                left_word, right_word = line_splitted[0:2]
                translation_candidates_list.append(((left_lang, left_word), (right_lang, right_word)))

        self.set_best_translation_candidates(translation_candidates_list)

    def get_word_by_vec(self, lang, vec):
        return self.models[lang].similar_by_vector(vec, topn=1)[0][0]

    def get_nearest_neighbors(self, lang, vec, n=100):
        return [(elem[0], elem[1]) for elem in self.models[lang].similar_by_vector(vec, topn=n)]

    def inflect_translation_pairs(self, lang1, words1, lang2, words2, treshold=0):
        words1 = set(words1)
        words2 = set(words2)

        translations1 = self.translate_default_set(lang1, lang2, words1)
        translations2 = self.translate_default_set(lang2, lang1, words2)

        lang1_to_lang2_pairs = self._intersect_translations(translations1, words1, words2)
        pairs1_len = len(lang1_to_lang2_pairs)
        lang2_to_lang1_pairs = self._intersect_translations(translations2, words2, words1)
        pairs2_len = len(lang2_to_lang1_pairs)

        lang1_to_lang2_sets = {}

        word2cluster = {"kir": {}, "kaz": {}}
        cluster_id = 0
        for idx, entry in enumerate(lang1_to_lang2_pairs):
            if idx % 500 == 0:
                self.logger.info(f"inflect_translation_pairs():{lang1} clustered {idx} pair of {pairs1_len}")
            if entry[2] > treshold:
                if entry[0] not in lang1_to_lang2_sets.keys():
                    lang1_to_lang2_sets[entry[0]] = {(entry[0], lang1)}

                if entry[0] not in word2cluster[lang1].keys():
                    word2cluster[lang1][entry[0]] = []
                word2cluster[lang1][entry[0]].append(cluster_id)

                if entry[1] not in word2cluster[lang2].keys():
                    word2cluster[lang2][entry[1]] = []
                word2cluster[lang2][entry[1]].append(cluster_id)

                lang1_to_lang2_sets[entry[0]].add((entry[1], lang2))

            cluster_id += 1

        for idx, entry in enumerate(lang2_to_lang1_pairs):
            if idx % 500 == 0:
                self.logger.info(f"inflect_translation_pairs():{lang2} clustered {idx} pair of {pairs2_len}")
            if entry[2] > treshold:
                # entry[1] cause we use lang1 as cluster naming language
                if entry[1] not in lang1_to_lang2_sets.keys():
                    lang1_to_lang2_sets[entry[1]] = {(entry[1], lang1)}

                if entry[0] not in word2cluster[lang2].keys():
                    word2cluster[lang2][entry[0]] = []
                word2cluster[lang2][entry[0]].append(cluster_id)

                if entry[1] not in word2cluster[lang1].keys():
                    word2cluster[lang1][entry[1]] = []
                word2cluster[lang1][entry[1]].append(cluster_id)

                lang1_to_lang2_sets[entry[1]].add((entry[0], lang2))

            cluster_id += 1

        return lang1_to_lang2_pairs, lang2_to_lang1_pairs, lang1_to_lang2_sets, word2cluster

    @staticmethod
    # todo: remove words_src cause it duplicates translations_src.keys()?
    def _intersect_translations(translations_src, words_src, words_tgt):
        lang_src_2_lang_tgt_pairs = []

        for word in words_src:
            for tr in translations_src[word]:
                if tr[1] in words_tgt:
                    lang_src_2_lang_tgt_pairs.append((word, tr[1], tr[2]))

        return lang_src_2_lang_tgt_pairs

    def translate_default_set(self, src_lang, tgt_lang, words):
        translations = {}
        words_len = len(words)
        for idx, word in enumerate(words):
            translations[word] = self.translate_default(src_lang, tgt_lang, word)
            if idx % 500 == 0:
                self.logger.info(f"translate_default_set():{src_lang} preprocessed {idx} words of {words_len}")
        return translations

    def _get_translations_of_translations(self, lang1, lang2, word1_translations):
        backtranslations = [self.translate_default(lang1, lang2, src_w_tr[1]) for src_w_tr in word1_translations]
        backtranslations = [item for sublist in backtranslations for item in sublist]
        return backtranslations

    def translate_default(self, src_lang, tgt_lang, src_word, n=10):
        return self._translate_naive(src_lang, tgt_lang, src_word, n=n)

    def initialize_translations(self):
        self._initialize_translations(self.lang1, self.lang2)
        self._initialize_translations(self.lang2, self.lang1)

    def _initialize_translations(self, src_lang, tgt_lang):
        i = 0
        l = len(self.models[src_lang].vocab)
        for wf in self.models[src_lang].vocab:
            if i % 1000 == 0:
                print(f"{i} of {l}")
            self.translations[src_lang][wf] = [(elem[1], elem[2]) for elem in
                                               self.translate_default(src_lang, tgt_lang, wf)]
            i += 1
