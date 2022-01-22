from typing import List, TextIO, Dict
from urllib import request, error
import os
from bs4 import BeautifulSoup
import re


path_data = "C:/Users/nicolas.bourgeois/Desktop/Backup/Recherche/articles/puren_methodo/methodo"
revue = "cb328020951"


def load_notices(start_year: int, end_year: int) -> None:
    print("loading notices")
    for year in range(start_year, end_year+1):
        xml_notices = f"https://gallica.bnf.fr/services/Issues?ark=ark:/12148/{revue}/date&date={year}"
        print(xml_notices)
        if not os.path.isfile(os.path.join(path_data, "notices", f"notices_{year}.xml")):

            try:
                distant_file = request.urlopen(xml_notices)
                xml_document: str = distant_file.read()
                file_notices = open(os.path.join(path_data, "notices", f"notices_{year}.xml"), "wb")
                file_notices.write(xml_document)
            except error.HTTPError as err:
                print(err)
        else:
            print("Année déjà traitée")
    print("notices loaded")


def load_xmlfiles() -> List[str]:
    print("building list of files")
    notices: List[str] = list()
    for notices_file_adr in os.listdir(os.path.join(path_data, "notices")):
        notices_file: TextIO = open(os.path.join(path_data, "notices", notices_file_adr), "r", encoding="utf-8")
        content_notices: str = notices_file.read()
        notices_file.close()
        notices += re.findall("ark=\"[a-z0-9]+\"", content_notices)
    notices = [notice[5:-1] for notice in notices]
    print("list built")
    return notices


def extract(num_issue: int, ark: str) -> None:
    img_source_jo: str = f"ark:/12148/{ark}.texteBrut"
    print(img_source_jo)
    target_url = f"https://gallica.bnf.fr/{img_source_jo}"
    print(target_url)
    try:
        distant_file = request.urlopen(target_url)
        html_document: str = distant_file.read()
        html_content_texte: str = BeautifulSoup(html_document, 'html.parser').get_text()
        head_vs_main: List[str] = re.split("Le taux de reconnaissance estimé pour ce document est de [0-9]+%",
                                           html_content_texte)

        local_file_texte: TextIO = open(os.path.join(path_data, "ocr", f"texte_{num_issue}.html"), "w", encoding="utf-8")
        local_file_metadata: TextIO = open(os.path.join(path_data, "ocr", f"metadata_{num_issue}.html"), "w")
        local_file_texte.write(head_vs_main[1])
        local_file_metadata.write(head_vs_main[0])
        local_file_texte.close()
        local_file_metadata.close()
    except error.HTTPError as err:
        print(err)


if __name__ == "__main__":
    load_notices(1881, 1940)
    all_year_notices: List[str] = load_xmlfiles()
    for index, numero in enumerate(all_year_notices):
        if not os.path.isfile(os.path.join(path_data, "ocr", f"texte_{index}.html")):
            extract(index, numero)
        else:
            print(f"{index} déjà traité")


