import pickle
from math import sqrt
from scipy.cluster.hierarchy import dendrogram
from statistics import mean
from typing import Dict, List, TextIO, Tuple, Set
import minisom
import pandas as pd
from matplotlib import pyplot as plt, cm
from matplotlib import dates as mdates
from datetime import datetime, date
from pytesseract import image_to_string, pytesseract
import cv2
import os
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.preprocessing import MinMaxScaler
from extraction_signal import moving_average

pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
path_data = "C:\\Users\\nicolas.bourgeois\\Desktop\\Backup\\Recherche\\articles\\puren_methodo\\methodo"


def ngram(forms: List[str]):
    plt.figure(figsize=(15, 10))
    comptage: Dict[str, List[int]] = {word: list() for word in forms}
    dates: List[date] = list()
    for session_file_adr in os.listdir(os.path.join(path_data, "comptage")):
        dates.append(datetime.strptime(session_file_adr[:-4], '%Y-%m-%d').date())
        session_compteur = pd.read_csv(os.path.join(path_data, "comptage", session_file_adr),
                                       sep=";", encoding="utf-8").set_index("Unnamed: 0")
        taille: int = session_compteur.sum()
        for word in forms:
            comptage[word].append(session_compteur.loc[word] / taille if word in session_compteur.index else 0)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=2))
    for word in forms:
        plt.plot(dates[18:-18], moving_average(comptage[word], 37), label=word)
    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.savefig(os.path.join(path_data, "graphes", "frequences.png"))
    plt.close()


def test_ocr(img_file: str) -> None:
    img = cv2.imread(os.path.join(path_data, "img", img_file))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(path_data, "img", f"gray_{img_file}"), gray)
    png = cv2.imread(os.path.join(path_data, "img", f"gray_{img_file}"))
    text = image_to_string(png, config="-l fra")
    print(text[:2000])


def cha_freq(export: bool) -> Tuple[AgglomerativeClustering, AgglomerativeClustering]:
    print("classification des fréquences")
    matx_occurrences: pd.DataFrame = pd.read_csv(os.path.join(path_data, "comptage_1882_corr.csv"), sep=";",
                                                 encoding="utf-8", header=0).set_index("Unnamed: 0")
    matx_occurrences.loc[:, "total"] = matx_occurrences.sum(axis=1)
    matx_occurrences = matx_occurrences.sort_values(by="total", ascending=False)
    matx_occurrences = matx_occurrences.loc[matx_occurrences.total > 50, :].drop("total", axis=1).fillna(0)
    if export:
        matx_cpy: pd.DataFrame = matx_occurrences.copy()
        matx_cpy = pd.DataFrame(MinMaxScaler().fit_transform(matx_cpy), index=matx_cpy.index, columns=matx_cpy.columns)
        print(matx_cpy.head())
        distances: pd.DataFrame = pd.DataFrame(np.zeros((len(matx_cpy.columns), len(matx_cpy.columns))),
                                               columns=matx_cpy.columns, index=matx_cpy.columns)
        for i, data1 in enumerate(distances.columns):
            matches = [co for j, co in enumerate(distances.columns) if j > i]
            for data2 in matches:
                x = sqrt(matx_cpy.loc[:, data1].subtract(matx_cpy.loc[:, data2]).apply(lambda z: z ** 2).sum())
                distances.loc[data1, data2] = x
                distances.loc[data2, data1] = x
        print(distances.head())
        distances.to_csv(os.path.join(path_data, "distances_1882.csv"), sep=";", encoding="utf-8")

    matx_occurrences = matx_occurrences.transpose()
    proximity = MinMaxScaler().fit_transform(matx_occurrences)
    distances = np.vectorize(lambda x: 1 - x)(proximity)
    print(distances.shape)
    np.savetxt("distances.csv", distances, delimiter=";")

    print("calcul CHA")
    n_clusters_w = 12
    n_clusters_c = 8
    cha: AgglomerativeClustering = AgglomerativeClustering(n_clusters=n_clusters_w,
                                                           linkage="ward", compute_distances=True)
    cha.fit(distances)
    cha2: AgglomerativeClustering = AgglomerativeClustering(n_clusters=n_clusters_c,
                                                            linkage="complete", compute_distances=True)
    cha2.fit(distances)
    print(len(cha.labels_))
    clusters: List[List[int]] = [[x] for x in range(len(cha.labels_))]
    is_active_cluster: List[bool] = [True for x in range(len(cha.labels_))]
    moyenne_dist_intra: List[int] = [0]
    size_biggest_cluster: List[int] = [0]
    max_dist_intra: List[int] = [0]
    num_iteration: int = 0
    f: TextIO = open(os.path.join(path_data, "cha_freq.txt"), "w", encoding="utf-8")
    while num_iteration < len(cha.labels_) - 1:
        feuille_1, feuille_2 = cha.children_[num_iteration]
        clusters.append(clusters[feuille_1] + clusters[feuille_2])
        is_active_cluster.append(True)
        is_active_cluster[feuille_1] = False
        is_active_cluster[feuille_2] = False
        active_clusters: List[List[int]] = [c for i, c in enumerate(clusters) if is_active_cluster[i]]
        f.write("\n************")
        f.write(str(num_iteration))
        f.write("\n")
        f.write("\n".join([f"{i} {c}" for i, c in enumerate(active_clusters)]))
        moyenne_dist_intra.append(max([0 if len(c) == 1 else mean(distances[i, j]
                                                                  for i in c for j in c if j < i)
                                       for c in active_clusters]))
        max_dist_intra.append(max([0 if len(c) == 1 else max(distances[i, j]
                                                             for i in c for j in c if j < i) for c in active_clusters]))
        size_biggest_cluster.append(max([len(c) for c in active_clusters]))
        num_iteration += 1
    f.close()
    plt.figure(figsize=(12, 8))
    plt.plot(range(len(cha2.distances_)), cha2.distances_, label="complete linkage")
    plt.plot(range(len(cha.distances_)), cha.distances_, label="Ward linkage")
    ymax: float = min(60, max([cha2.distances_[-1], cha.distances_[-1]]))
    plt.annotate(s=f"{n_clusters_w} classes", xy=(len(cha.distances_) - n_clusters_w - 25, 2))
    plt.vlines(len(cha.distances_) - n_clusters_w, 0, ymax, linestyles="dashed", color="orange", linewidth=2)
    plt.annotate(s=f"{n_clusters_c} classes", xy=(len(cha.distances_) - n_clusters_c + 1, 2))
    plt.vlines(len(cha.distances_) - n_clusters_c, 0, ymax, linestyles="dashed", color="blue", linewidth=2)
    plt.ylim(0, ymax)
    plt.grid()
    plt.legend()
    plt.xlabel("number of steps")
    plt.savefig(os.path.join(path_data, "cha_frq_dist.png"))
    plt.close()
    return cha, cha2


def acp_freq(cha: AgglomerativeClustering, cha2: AgglomerativeClustering):
    matx_occurrences: pd.DataFrame = pd.read_csv(os.path.join(path_data, "comptage_1882_corr.csv"), sep=";",
                                                 encoding="utf-8", header=0).set_index("Unnamed: 0")
    matx_occurrences.loc[:, "total"] = matx_occurrences.sum(axis=1)
    matx_occurrences = matx_occurrences.sort_values(by="total", ascending=False)
    matx_occurrences = matx_occurrences.loc[matx_occurrences.total > 50, :].drop("total", axis=1).fillna(0)
    matx_occurrences = matx_occurrences.transpose()
    proximity = MinMaxScaler().fit_transform(matx_occurrences)
    print("calcul ACP")
    pca = PCA()
    XP = pca.fit_transform(proximity)
    print(XP.shape)
    print("variance expliquée", pca.explained_variance_ratio_)
    features: List[str] = [date[5:] for date in matx_occurrences.index]
    print("features", features)
    for axe in range(4):
        plt.figure(figsize=(14, 14))
        plt.scatter(XP[:, 2 * axe], XP[:, 2 * axe + 1], edgecolors="none", marker="o", c=cha.labels_, cmap="tab10",
                    s=60)
        for j, feature in enumerate(features):
            if feature in ["02-23", "05-11", "02-13", "02-23_1", "05-11_1", "02-13_1"]:
                plt.annotate(xy=(XP[j, 2 * axe] + 0.01, XP[j, 2 * axe + 1]), s=feature)
        plt.savefig(os.path.join(path_data, f"acp_freq_axes_{2 * axe}_{2 * axe + 1}.png"))


def som_freq(cha: AgglomerativeClustering, cha2: AgglomerativeClustering):
    matx_occurrences: pd.DataFrame = pd.read_csv(os.path.join(path_data, "comptage_1882_corr.csv"), sep=";",
                                                 encoding="utf-8", header=0).set_index("Unnamed: 0")
    matx_occurrences.loc[:, "total"] = matx_occurrences.sum(axis=1)
    matx_occurrences = matx_occurrences.sort_values(by="total", ascending=False)
    matx_occurrences = matx_occurrences.loc[matx_occurrences.total > 50, :].drop("total", axis=1).fillna(0)
    matx_occurrences = matx_occurrences.transpose()
    proximity = MinMaxScaler().fit_transform(matx_occurrences)
    features: List[str] = [date[5:] for date in matx_occurrences.index]
    print("calcul SOM")
    n_neurons: int = 14
    m_neurons: int = 14
    som: minisom.MiniSom = minisom.MiniSom(n_neurons, m_neurons, proximity.shape[1], sigma=5, learning_rate=.5,
                                           neighborhood_function='gaussian')
    som.pca_weights_init(proximity)
    som.train(proximity, 2000, verbose=True)
    with open(os.path.join(path_data, "SOM_freq_clusters.p"), 'wb') as outfile:
        pickle.dump(som, outfile)

    with open(os.path.join(path_data, "SOM_freq_clusters.p"), 'rb') as infile:
        som: minisom.MiniSom = pickle.load(infile)
    distances_map = som.distance_map()
    print(distances_map)
    cases: Dict[Tuple[int, int], List[int]] = {(xn, yn): list() for xn in range(n_neurons) for yn in range(m_neurons)}
    for num_data, xx in enumerate(proximity):
        cases[som.winner(xx)].append(num_data)
    print(cases)
    plt.figure(figsize=(14, 14))
    plt.xlim(0, n_neurons)
    plt.ylim(0, m_neurons)
    plt.yticks([])
    plt.xticks([])
    cmap = cm.get_cmap('tab10')
    for case in cases:
        plt.scatter(x=case[0] + 0.5, y=case[1] + 0.5, marker="s", s=2600, c=str(distances_map[case]))
        if len(cases[case]) > 0:
            for axe, t in enumerate(cases[case]):
                plt.annotate(xy=(case[0] + .15, case[1] + .8 - axe / 5), s=features[t], fontsize="small",
                             color="black" if distances_map[case] > 0.5 else "white", fontweight="heavy")
    plt.savefig(os.path.join(path_data, "som_freq.png"))
    plt.figure(figsize=(12, 12))
    plt.xlim(0, n_neurons)
    plt.ylim(0, m_neurons)
    plt.yticks([])
    plt.xticks([])
    cmap = cm.get_cmap('tab10')
    for case in cases:
        plt.scatter(x=case[0] + 0.5, y=case[1] + 0.5, marker="s", s=2600, c=str(distances_map[case]))
        if len(cases[case]) > 0:
            for axe, t in enumerate(cases[case]):
                print(features[t], cha.labels_[t])
                plt.annotate(xy=(case[0] + .15, case[1] + .8 - axe / 5), s=features[t], fontsize="small",
                             color=cmap(cha.labels_[t]), fontweight="heavy")
    plt.savefig(os.path.join(path_data, "som_freq_with_cha.png"))


def acp_per_year():
    for year in range(1881, 1941):
        print(year)
        matx_occurrences: pd.DataFrame = pd.read_csv(os.path.join(path_data, "comptage_annees", f"comptage_{year}.csv"),
                                                     sep=";", encoding="utf-8", header=0).set_index("Unnamed: 0")
        matx_occurrences.loc[:, "total"] = matx_occurrences.sum(axis=1)
        matx_occurrences = matx_occurrences.sort_values(by="total", ascending=False)
        matx_occurrences = matx_occurrences.loc[matx_occurrences.total > 50, :].drop("total", axis=1).fillna(0)
        matx_occurrences = matx_occurrences.transpose()
        proximity = MinMaxScaler().fit_transform(matx_occurrences)
        distances = np.vectorize(lambda x: 1 - x)(proximity)

        cha: AgglomerativeClustering = AgglomerativeClustering(n_clusters=2, linkage="ward")
        cha.fit(distances)
        cha_colors = ["orange" if l == 0 else "green" for l in cha.labels_]
        km: KMeans = KMeans(n_clusters=2)
        km.fit(distances)
        km_colors = ["orange" if l == 0 else "green" for l in km.labels_]

        pca = PCA()
        XP = pca.fit_transform(proximity)
        print("variance expliquée", pca.explained_variance_ratio_)
        plt.figure(figsize=(14, 14))
        plt.scatter(XP[:, 0], XP[:, 1], edgecolors="none", marker="o", cmap="tab10", s=60, c=cha_colors)
        plt.savefig(os.path.join(path_data, "acps", f"acp_{year}_cha.png"))
        plt.figure(figsize=(14, 14))
        plt.scatter(XP[:, 0], XP[:, 1], edgecolors="none", marker="o", cmap="tab10", s=60, c=km_colors)
        plt.savefig(os.path.join(path_data, "acps", f"acp_{year}_km.png"))
        plt.close()


def carto_coocs() -> None:
    print("classification des cooccurences")
    matx_coocurences: pd.DataFrame = pd.read_csv(os.path.join(path_data, "coocs.csv"),
                                                 sep=";", encoding="utf-8", header=0).set_index("word")
    proximity = MinMaxScaler().fit_transform(matx_coocurences)
    distances = np.vectorize(lambda x: 1 - x)(proximity)
    print(distances)

    print("calcul CHA")
    cha = AgglomerativeClustering(n_clusters=8)
    cha.fit(distances)
    print(len(cha.labels_))
    clusters: List[List[int]] = [[x] for x in range(len(cha.labels_))]
    is_active_cluster: List[bool] = [True for x in range(len(cha.labels_))]
    moyenne_dist_intra: List[int] = [0]
    size_biggest_cluster: List[int] = [0]
    max_dist_intra: List[int] = [0]
    num_iteration: int = 0
    f: TextIO = open(os.path.join(path_data, "cha.txt"), "w", encoding="utf-8")
    while num_iteration < len(cha.labels_) - 1:
        feuille_1, feuille_2 = cha.children_[num_iteration]
        clusters.append(clusters[feuille_1] + clusters[feuille_2])
        is_active_cluster.append(True)
        is_active_cluster[feuille_1] = False
        is_active_cluster[feuille_2] = False
        active_clusters: List[List[int]] = [c for i, c in enumerate(clusters) if is_active_cluster[i]]
        f.write("\n************")
        f.write(str(num_iteration))
        f.write("\n")
        f.write("\n".join([f"{i} {c}" for i, c in enumerate(active_clusters)]))
        moyenne_dist_intra.append(max([0 if len(c) == 1 else mean(distances[i, j]
                                                                  for i in c for j in c if j < i)
                                       for c in active_clusters]))
        max_dist_intra.append(max([0 if len(c) == 1 else max(distances[i, j]
                                                             for i in c for j in c if j < i) for c in active_clusters]))
        size_biggest_cluster.append(max([len(c) for c in active_clusters]))
        num_iteration += 1
    f.close()

    plt.figure(figsize=(12, 8))
    plt.plot(range(len(moyenne_dist_intra)), max_dist_intra, label="pire distance intra")
    plt.plot(range(len(moyenne_dist_intra)), size_biggest_cluster, label="plus gros cluster")
    plt.grid()
    plt.legend()
    plt.xlabel("nombre d'étapes")
    plt.savefig(os.path.join(path_data, "cha_dist.png"))
    plt.close()

    print("calcul ACP")
    pca = PCA(n_components=8)
    XP = pca.fit_transform(proximity)
    print("variance expliquée", pca.explained_variance_ratio_)
    features: List[str] = matx_coocurences.columns
    print("features", features)
    for axe in range(4):
        plt.figure(figsize=(10, 10))
        plt.scatter(XP[:, 2 * axe], XP[:, 2 * axe + 1], edgecolors="none", marker="o", c=cha.labels_, cmap="tab10",
                    s=60)
        for j, feature in enumerate(features):
            plt.annotate(xy=(XP[j, 2 * axe] + 0.01, XP[j, 2 * axe + 1]), s=feature)
        plt.savefig(os.path.join(path_data, f"acp_axes_{2 * axe}_{2 * axe + 1}.png"))

    print("calcul SOM")
    n_neurons: int = 8
    m_neurons: int = 8
    som = minisom.MiniSom(n_neurons, m_neurons, proximity.shape[1], sigma=1.5, learning_rate=.5,
                          neighborhood_function='gaussian', random_seed=0)
    som.pca_weights_init(proximity)
    som.train(proximity, 2000, verbose=True)
    with open(os.path.join(path_data, "SOM_clusters.p"), 'wb') as outfile:
        pickle.dump(som, outfile)

    with open(os.path.join(path_data, "SOM_clusters.p"), 'rb') as infile:
        som = pickle.load(infile)
    cases: Dict[Tuple[int, int], List[int]] = {(xn, yn): list() for xn in range(n_neurons) for yn in range(m_neurons)}
    for num_data, xx in enumerate(proximity):
        cases[som.winner(xx)].append(num_data)
    print(cases)
    plt.figure(figsize=(12, 12))
    plt.xlim(0, n_neurons)
    plt.ylim(0, m_neurons)
    plt.yticks([])
    plt.xticks([])
    cmap = cm.get_cmap('tab10')
    for case in cases:
        if len(cases[case]) > 0:
            for axe, t in enumerate(cases[case]):
                plt.annotate(xy=(case[0] + .15, case[1] + .7 - axe / 4), s=features[t], fontsize="small",
                             color=cmap(cha.labels_[t] / 10), fontweight="heavy")
    plt.savefig(os.path.join(path_data, "som.png"))


def test_comptage(mots: List[str], cha1: AgglomerativeClustering, cha2: AgglomerativeClustering):
    comptage_1882: pd.DataFrame = pd.read_csv(os.path.join(path_data, "comptage_1882_corr.csv"), sep=";",
                                              encoding="utf-8", header=0).set_index("Unnamed: 0").transpose()
    indice = {text: num for num, text in enumerate(comptage_1882.index.to_list())}
    print(indice)
    comptage_1882.loc[:, "c1"] = comptage_1882.index.map(lambda z: cha1.labels_[indice[z]])
    comptage_1882.loc[:, "c2"] = comptage_1882.index.map(lambda z: cha2.labels_[indice[z]])
    for mot in mots:
        best_choice: pd.DataFrame = comptage_1882.sort_values(by=mot, ascending=False).loc[:, [mot, "c1", "c2"]]
        print(best_choice.head(10))


def topic_models(from_rep: str, to_rep: str) -> None:
    nb_topics: int = 100
    words_per_topic: int = 50
    print("Topic modeling")
    compteurs_par_annee: Dict[str, pd.DataFrame] = dict()
    for annee in os.listdir(os.path.join(path_data, from_rep)):
        print(annee)
        Xloc: pd.DataFrame = pd.read_csv(os.path.join(path_data, from_rep, annee), sep=";", encoding='utf-8', header=0,
                                         index_col=0)
        Xloc.loc[:, "total"] = Xloc.sum(axis=1)
        compteurs_par_annee[annee] = Xloc.loc[Xloc.total > 50].drop("total", axis=1).transpose()
    X: pd.DataFrame = pd.concat(compteurs_par_annee, join="outer", axis=0).fillna(0)
    print(X.shape)
    print(X.info())
    clefs: List[str] = list(X.columns)
    pages: List[str] = list(X.index)
    lda = LatentDirichletAllocation(n_components=nb_topics)
    topic_to_text: np.ndarray[float, float] = lda.fit_transform(X)
    table_topics_to_texts: pd.DataFrame = pd.DataFrame(np.vectorize(lambda z: f"{z:.3f}")(topic_to_text),
                                                       columns=range(nb_topics), index=pages)
    table_topics_to_texts.to_csv(os.path.join(path_data, to_rep, f"blocs_texts_toptotext.csv"),
                                 encoding="utf-8", sep=";", index=True)

    topics: pd.DataFrame = pd.DataFrame({f"Topic{i}": [clefs[w] for w in top.argsort()[-words_per_topic:]]
                                         for i, top in enumerate(lda.components_)})
    topics.to_csv(os.path.join(path_data, to_rep, f"blocs_textes_topics.csv"), encoding="utf-8", sep=";", index=False)

    print(f"TM terminé")


def topics_proxy(rep: str):
    print("Calcul de la matrice de proximité")
    comptage = pd.read_csv(os.path.join(path_data, rep, "blocs_textes_topics.csv"), sep=";", encoding="utf-8", header=0,
                           index_col=None)
    topic_sets: List[str] = list(comptage.columns)
    nb_topics: int = len(topic_sets)
    proximity: np.ndarray = np.array([
        [len(set(comptage.loc[:, f"Topic{nt1}"].to_list()) & set(comptage.loc[:, f"Topic{nt2}"].to_list()))
         for nt1 in range(nb_topics)] for nt2 in range(nb_topics)])
    np.savetxt(os.path.join(path_data, rep, "prox.csv"), proximity, delimiter=",")
    print("Calcul terminé")


def topics_cha(rep: str) -> AgglomerativeClustering:
    print("Classification hiérarchique")
    words_in_topic: int = 50
    distance_max: int = 25
    proximity: np.ndarray = np.loadtxt(os.path.join(path_data, rep, "prox.csv"), delimiter=",")
    distances: np.ndarray = np.vectorize(lambda x: words_in_topic - x)(proximity)

    cha = AgglomerativeClustering(affinity="precomputed", linkage="complete",
                                  distance_threshold=distance_max, n_clusters=None)
    cha.fit(distances)
    print(cha.labels_)
    plt.figure(figsize=(12, 12), dpi=300)
    counts = np.zeros(cha.children_.shape[0])
    n_samples = len(cha.labels_)
    for i, merge in enumerate(cha.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack([cha.children_, cha.distances_, counts]).astype(float)
    dendrogram(linkage_matrix, color_threshold=distance_max)
    plt.savefig(os.path.join(path_data, rep, "dendogram_cha.png"))
    return cha


if __name__ == "__main__":
    # ngram(["communiste", "ordre", "boulanger", "eglise", "guerre"])
    # test_ocr("jo33.jpg")
    # cha_w, cha_c = cha_freq(export=False)
    # test_comptage(["tunisie", "guerre", "algérie", "armée", "militaire", "colonie"], cha_w, cha_c)
    # acp_freq(cha_w, cha_c)
    # som_freq(cha_w, cha_c)
    # acp_per_year()
    # topic_models("comptage_annees", "topics")
    # topics_proxy("topics")
    cha_topics = topics_cha("topics")
