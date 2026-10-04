from candidate_rag.core import KeywordSearch, Passage, fuse, tokens


def test_page_provenance_and_keyword_search():
    passages = [Passage("guide.pdf", 3, 1, "The library closes at midnight."),
                Passage("calendar.pdf", 2, 1, "The bus leaves at seven.")]
    hits = KeywordSearch(passages).search("library midnight")
    assert hits[0][0].key == "guide.pdf:p3:c1"
    assert fuse(hits, []) == [passages[0]]


def test_chinese_bigrams():
    assert "图书" in tokens("图书馆开放")
    assert KeywordSearch([Passage("a.pdf", 1, 1, "图书馆开放")]).search("图书馆")
