from gensim.models import KeyedVectors
import logging
import sys


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

    def meaning_clustering(self, lang1, word1, lang2, word2, depth=2):

        # print(self.models[lang1])
        self.logger.info("BEFORE meaning_clustering()")

        word1_translations = self.translate_default(lang1, lang2, word1)
        word2_translations = self.translate_default(lang2, lang1, word2)
        if depth == 2:
            word1_backtranslations = self._get_translations_of_translations(lang2, lang1, word1_translations)
            word2_backtranslations = self._get_translations_of_translations(lang1, lang2, word2_translations)

            word1_backbacktranslations = self._get_translations_of_translations(lang1, lang2, word1_backtranslations)
            word2_backbacktranslations = self._get_translations_of_translations(lang2, lang1, word2_backtranslations)

        w1_l2_tr_1 = set([tr[1] for tr in word1_translations])
        if depth == 2:
            w1_l2_tr_2 = set([tr[1] for tr in word1_backbacktranslations])
            w1_l1_tr_1 = set([tr[1] for tr in word1_backtranslations])

        w2_l1_tr_1 = set([tr[1] for tr in word2_translations])
        if depth == 2:
            w2_l1_tr_2 = set([tr[1] for tr in word2_backbacktranslations])
            w2_l2_tr_1 = set([tr[1] for tr in word2_backtranslations])

        #todo naming!!!
        int1 = w1_l2_tr_1.intersection({word2})
        int2 = w1_l2_tr_1.intersection(w2_l2_tr_1) if depth == 2 else set()
        int3 = w1_l2_tr_2.intersection({word2}) if depth == 2 else set()
        int4 = w1_l2_tr_2.intersection(w2_l2_tr_1) if depth == 2 else set()

        int5 = w2_l1_tr_1.intersection({word1})
        int6 = w2_l1_tr_1.intersection(w1_l1_tr_1) if depth == 2 else set()
        int7 = w2_l1_tr_2.intersection({word1}) if depth == 2 else set()
        int8 = w2_l1_tr_2.intersection(w1_l1_tr_1) if depth == 2 else set()

        sizes = {
            "depth 0": [
                {f"{lang1}2{lang2}": len(int1), f"{lang2}2{lang1}": len(int5)},
                {f"{lang1}2{lang2}": len(int3), f"{lang2}2{lang1}": len(int7)}],
            "depth 1": {f"{lang1}2{lang2}": len(int2), f"{lang2}2{lang1}": len(int6)},
            "depth 2": {f"{lang1}2{lang2}": len(int4), f"{lang2}2{lang1}": len(int8)}
        }

        self.logger.info("AFTER meaning_clustering()")

        return sizes


    def share_meaning(self, lang1, word1, lang2, word2, depth=2):
        meaning_clustering = self.meaning_clustering(lang1, word1, lang2, word2, depth)

        not_share_meaning = 0 in meaning_clustering["depth 0"][0].values() and \
                            0 in meaning_clustering["depth 0"][1].values() and \
                            0 in meaning_clustering["depth 1"].values() and \
                            0 in meaning_clustering["depth 2"].values()

        return not not_share_meaning


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
        for wf in self.models[src_lang].vocab:
            self.translations[src_lang][wf] = [(elem[1], elem[2]) for elem in self.translate_default(src_lang, tgt_lang, wf)]
