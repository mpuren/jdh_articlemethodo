import os
import shutil
from typing import TextIO, List, Dict, Tuple, Set
import re
from nltk import word_tokenize
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer as FLF
from collections import Counter
import pandas as pd
import numpy as np

path_data = "C:\\Users\\nicolas.bourgeois\\Desktop\\Backup\\Recherche\\articles\\puren_methodo\\methodo"


def dictionnaires() -> Tuple[Set[str], Set[str], Set[str], Set[str], Set[str]]:
    frenchwords: List[str] = open(os.path.join(path_data, "french_words.txt"), "r",
                                  encoding="utf-8").read().split("\n")
    frenchwords = [w.strip() for w in frenchwords]
    stopwords = open(os.path.join(path_data, "french_stopwords.txt"), "r", encoding="utf-8").read().split("\n")
    stopwords = [w.strip() for w in stopwords]
    lemmatizer = FLF()
    lems: List[str] = [lemmatizer.lemmatize(w) for w in frenchwords]
    stop_lems: List[str] = [lemmatizer.lemmatize(w) for w in stopwords]
    propres: List[str] = open(os.path.join(path_data, "homonymes_propres.txt"), "r",
                              encoding="utf-8").read().split("\n")
    propres: List[str] = [w.strip() for w in propres]
    return set(frenchwords), set(lems), set(stopwords), set(stop_lems), set(propres)


def moving_average(data: List[float], periode: int):
    return np.convolve(np.array(data), np.ones(periode), 'valid') / periode


def comptage_par_date() -> None:
    for session_file_adr in os.listdir(os.path.join(path_data, "ocr")):
        if session_file_adr[:8] == "metadata":
            root = session_file_adr[8:]
            metadata_file: TextIO = open(os.path.join(path_data, "ocr", session_file_adr), encoding="ANSI")
            metadata_text = metadata_file.read()
            metadata_file.close()
            body_file: TextIO = open(os.path.join(path_data, "ocr", "texte" + root), encoding="utf-8")
            body_text = " ".join(re.findall("[a-zâêûîôäëüïöùàçéè-]+", body_file.read().lower()))
            body_file.close()

            infos_dates: str = re.findall("Date d'édition : [0-9\-]+", metadata_text)[0]
            timing: str = infos_dates.split(":")[1][1:]
            bag_of_words: Dict[str, str] = Counter(word_tokenize(body_text, language="french"))
            distribution: pd.Series = pd.Series(bag_of_words)
            distribution.to_csv(os.path.join(path_data, "comptage", f"{timing}.csv"), encoding="utf-8", sep=";")


def sort_years(ocr: str, sorted: str):
    root_filenames = [fi[8:] for fi in os.listdir(os.path.join(path_data, ocr)) if fi[:8] == "metadata"]
    for rfn in root_filenames:
        meta_file: TextIO = open(os.path.join(path_data, ocr, f"metadata{rfn}"), "r", encoding="ANSI")
        meta_content: str = meta_file.read()
        meta_file.close()
        date: str = re.findall(r"Date d'édition : [0-9\-]+", meta_content)[0][17:]
        compteur: int = 0
        if not os.path.exists(os.path.join(path_data, sorted, f"{date}.txt")):
            compteur = 0
            shutil.copy(os.path.join(path_data, ocr, f"texte{rfn}"), os.path.join(path_data, sorted, f"{date}.txt"))
        else:
            compteur += 1
            shutil.copy(os.path.join(path_data, ocr, f"texte{rfn}"),
                        os.path.join(path_data, sorted, f"{date}_{compteur}.txt"))


def compute_frequences(ocr: str, cmpt: str) -> None:
    print("\n***Counting starts***")
    for year in range(1881, 1941):
        print(f"Comptage année {year}")
        lemmatizer = FLF()
        year_data = [fi for fi in os.listdir(os.path.join(path_data, ocr)) if fi[:4] == str(year)]
        for txt_file in year_data:
            print(txt_file)
            if not os.path.isfile(os.path.join(path_data, cmpt, f"{txt_file[:-3]}csv")):
                table_frequences: pd.DataFrame = pd.DataFrame()
                filedata: TextIO = open(os.path.join(path_data, ocr, txt_file), "r", encoding="utf-8")
                texte_page: str = filedata.read()
                filedata.close()
                texte_page = " ".join(re.findall("[a-zâêûîôäëüïöùàçéè\-]+", texte_page.lower()))
                bag_of_words: List[str] = word_tokenize(texte_page.lower(), language="french")
                bag_of_words = [w for w in bag_of_words if 2 < len(w) < 22]
                bag_of_lems: List[str] = [lemmatizer.lemmatize(w) for w in bag_of_words]
                bag_of_lems = [w for w in bag_of_lems if w not in lemstopsfr]
                counter: Counter[str] = Counter(bag_of_lems)
                for key in counter:
                    table_frequences.loc[key, txt_file[:-4]] = counter[key]
                table_frequences.to_csv(os.path.join(path_data, cmpt, f"{txt_file[:-3]}csv"),
                                        sep=";", encoding="utf-8")
    print(f"***Counting ends***")


def combine_frequences(cmbt: str, cpl: str) -> None:
    for year in range(1881, 1941):
        print(f"Compilation année {year}")
        year_data = [fi for fi in os.listdir(os.path.join(path_data, cmbt)) if fi[:4] == str(year)]
        if not os.path.isfile(os.path.join(path_data, cpl, f"comptage_{year}.csv")):
            table_frequences: pd.DataFrame = pd.DataFrame()
            for txt_file in year_data:
                local_serie = pd.read_csv(os.path.join(path_data, cmbt, txt_file), sep=";", encoding="utf-8",
                                          index_col=0)
                try:
                    table_frequences = pd.concat([table_frequences, local_serie], axis=1, join="outer")
                except pd.errors.InvalidIndexError as er:
                    print(txt_file)
            print(table_frequences.head())
            table_frequences.to_csv(os.path.join(path_data, cpl, f"comptage_{year}.csv"), sep=";", encoding="utf-8")


def coocurences(ocr: str) -> None:
    print("calcul des coocurrences")
    topwords: pd.DataFrame = pd.read_csv(os.path.join(path_data, "comptage.csv"), sep=";", encoding="utf-8")
    topwords = topwords.loc[topwords.total > 40].rename({"Unnamed: 0": "word"}, axis=1)
    lemmatizer = FLF()
    cooc_matrix = pd.DataFrame(index=topwords.word, columns=topwords.word)
    for i, row in topwords.iterrows():
        print(row["word"])
        sentences: List[str] = list()
        table_coocs_word = pd.DataFrame(index=["total"])
        for numero_revue in os.listdir(os.listdir(os.path.join(path_data, ocr))):
            for txt_file in os.listdir(os.path.join(path_data, ocr, numero_revue)):
                page_texte = open(os.path.join(path_data, ocr, numero_revue, txt_file), "r", encoding="utf-8").read()
                sentences += re.findall(r"[^.!?\n]*[^A-Za-zàâäéèêëïîùüûôç]{}[^.!?\n]*".format(row["word"].lower()),
                                        page_texte)
                sentences += re.findall(r"[^.!?\n]*[^A-Za-zàâäéèêëïîùüûôç]{}[^.!?\n]*".format(row["word"].title()),
                                        page_texte)
                sentences += re.findall(r"[^.!?\n]*[^A-Za-zàâäéèêëïîùüûôç]{}[^.!?\n]*".format(row["word"].upper()),
                                        page_texte)
        cotext: str = " ".join(sentences)
        cotext = cotext.replace(r"'", " ")
        cotext = re.sub(r"[._<>~@]", "", cotext)
        bag_of_words: List[str] = word_tokenize(cotext.lower(), language="french")
        bag_of_words = [w for w in bag_of_words if 2 < len(w) < 22]
        bag_of_lems: List[str] = [lemmatizer.lemmatize(w) for w in bag_of_words]
        bag_of_lems = [w for w in bag_of_lems if w not in lemstopsfr]
        counter_bag_of_lems: Counter[str] = Counter(bag_of_lems)
        for key in counter_bag_of_lems:
            table_coocs_word.loc[key, "total"] = counter_bag_of_lems[key]
            if key in topwords.word.unique():
                cooc_matrix.loc[row["word"], key] = counter_bag_of_lems[key]
        table_coocs_word = table_coocs_word.sort_values(by="total", ascending=False)
        table_coocs_word = table_coocs_word.drop(row['word'], axis=0)
        table_coocs_word.to_csv(os.path.join(path_data, "coocs", f"{row['word']}.csv"), sep=";", encoding="utf-8")
    cooc_matrix = cooc_matrix.fillna(0)
    cooc_matrix.to_csv(os.path.join(path_data, "coocs.csv"), sep=";", encoding="utf-8")
    print("calcul terminé")


if __name__ == "__main__":
    motsfr, lemsfr, stopsfr, lemstopsfr, proprescommuns = dictionnaires()
    # comptage_par_date()
    # ngram(["communiste", "ordre", "boulanger", "eglise", "guerre"])
    # test_ocr("jo33.jpg")
    # sort_years("ocr", "ocr_sorted")
    compute_frequences("ocr_sorted", "comptage_lemmes")
    combine_frequences("comptage_lemmes", "comptage_annees")
