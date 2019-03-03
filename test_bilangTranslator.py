from unittest import TestCase
from translator import BilangTranslator


class TestFastBilangTranslator(TestCase):
    l1 = "kaz"
    l2 = "kir"
    e_fn1 = "aligned-kaz.vec.short"
    e_fn2 = "aligned-kir.vec.short"
    smallTranslator = BilangTranslator(l1, l2, e_fn1, e_fn2, initialize_translations=False)
    # def test__translate_naive(self):
    #     self.fail()
    #
    # def test_translate_naive(self):
    #     self.fail()
    #
    # def test_translate_morfessor_boosted(self):
    #     self.fail()
    #
    def test_get_vector(self):
        lang = "kir"
        word = "бир"
        print(self.smallTranslator.get_vec_by_word(lang, word))

    def test_get_word(self):
        lang = "kir"
        word = "бир"
        assert self.smallTranslator.get_word_by_vec(lang, self.smallTranslator.get_vec_by_word(lang, word)) == word

    def test_set_best_translation_candidates_from_tsv(self):
        self.smallTranslator.set_best_translation_candidates_from_tsv("kir", "kaz", "kir-kaz-top-translations.tsv")
        self.smallTranslator.get_best_translation_candidates()

    def test_get_nearest_neighbors(self):
        lang = "kir"
        word = "бир"
        self.smallTranslator.get_nearest_neighbors(lang, self.smallTranslator.get_vec_by_word(lang, word))

    def test_initialize_translations(self):
        self.smallTranslator.initialize_translations()
        pass




class TestLargeBilangTranslator(TestCase):
    l1 = "kaz"
    l2 = "kir"
    e_fn1 = "aligned-kaz.vec"
    e_fn2 = "aligned-kir.vec"
    largeTranslator = BilangTranslator(l1, l2, e_fn1, e_fn2)


    def test_meaning_clustering(self):
        lang1 = "kaz"
        word1 = "құда"
        lang2 = "kir"
        word2 = "куда"
        word22 = "заттын"
        self.largeTranslator.meaning_clustering(lang1, word1, lang2, word2)
        self.largeTranslator.meaning_clustering(lang1, word1, lang2, word22)

    def test_share_meaning(self):
        lang1 = "kaz"
        word1 = "құда"
        lang2 = "kir"
        word2 = "куда"
        word22 = "заттын"
        assert self.largeTranslator.share_meaning(lang1, word1, lang2, word2, depth=1)
        assert not self.largeTranslator.share_meaning(lang1, word1, lang2, word22)