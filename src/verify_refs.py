"""Verify every DOI in configs/references.yaml through Crossref (or DataCite for arXiv DOIs) and
store the metadata used to typeset the reference list (results/references_verified.json)."""
import os, json, time, urllib.request, urllib.parse
import yaml
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "entropy-benchmark/1.0 (mailto:akintsovns@gmail.com)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def crossref(doi):
    m = get("https://api.crossref.org/works/" + urllib.parse.quote(doi))["message"]
    yr = (m.get("published-print") or m.get("published-online") or m.get("issued"))["date-parts"][0][0]
    authors = [(a.get("family", ""), a.get("given", "")) for a in m.get("author", [])] or \
              [(a.get("family", ""), a.get("given", "")) for a in m.get("editor", [])]
    return dict(source="crossref", doi=m["DOI"], title=(m.get("title") or [""])[0], authors=authors, year=yr,
                container=(m.get("container-title") or [""])[0], short=(m.get("short-container-title") or [""])[0],
                volume=m.get("volume", ""), issue=m.get("issue", ""), page=m.get("page", ""),
                article_number=m.get("article-number", ""), type=m.get("type", ""), publisher=m.get("publisher", ""))


def datacite(doi):
    m = get("https://api.datacite.org/dois/" + urllib.parse.quote(doi))["data"]["attributes"]
    authors = [(c.get("familyName", c.get("name", "")), c.get("givenName", "")) for c in m.get("creators", [])]
    return dict(source="datacite", doi=m["doi"], title=m["titles"][0]["title"], authors=authors,
                year=m.get("publicationYear"), container=m.get("publisher", ""), short="", volume="", issue="",
                page="", article_number="", type=m.get("types", {}).get("resourceTypeGeneral", ""),
                publisher=m.get("publisher", ""))


def main():
    refs = yaml.safe_load(open(os.path.join(ROOT, "configs", "references.yaml")))
    out = {}
    for k, doi in refs.items():
        try:
            out[k] = (datacite if doi.lower().startswith("10.48550") else crossref)(doi); out[k]["verified"] = True
        except Exception as e:
            out[k] = dict(doi=doi, verified=False, error=str(e))
        print(k, out[k].get("verified"), out[k].get("year"), out[k].get("title", "")[:70], flush=True)
        time.sleep(0.3)
    json.dump(out, open(os.path.join(ROOT, "results", "references_verified.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
