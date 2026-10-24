# Reference verification (manuscript v7 → camera-ready)

Checked 2026-10-14. Each entry was compared with the publisher's or registry's own record: Crossref (DOI records), DataCite,
the PMLR / NeurIPS / MLSys / USENIX proceedings pages, arXiv, dblp, and the owners' pages for the two datasets that have no DOI.
Google Scholar could not be queried from the sandbox; **please spot-check each title there** (search the exact title in quotes) —
the reviewer asked for it, and it takes about ten minutes. Corrected BibTeX: `paper/refs_verified.bib`.

Reviewer 1, comments 7 and 8 name "references 7 and 12" without saying which version they read. In v6 those are IoT-23 (Zenodo,
real) and McMahan et al. 2017 (real). The entries that were actually wrong are listed first.

## Defects found in v7 (fix before submission)

| v7 | Problem | Correct entry | Evidence |
|---|---|---|---|
| [5] Booij et al. | Author list of one paper, title / journal / DOI of another. DOI 10.1016/j.scs.2021.102994 is **Moustafa (sole author), "A new distributed architecture … Network TON_IoT datasets", Sustainable Cities and Society 72, 102994 (2021)**. | Booij, Chiscop, Meeuwissen, Moustafa, den Hartog: "ToN_IoT: The role of heterogeneity and the need for standardization of features and attack types in IoT network intrusion data sets", IEEE Internet of Things Journal 9(1), 485–496 (2022), DOI 10.1109/JIOT.2021.3085194. Keep the Moustafa 2021 paper as its own reference (it is the descriptor of the *network* ToN_IoT files this benchmark uses). | Crossref for both DOIs |
| [23] Yin et al. | Authors are those of a different paper ("Gradient diversity…", Yin, Pananjady, Lam, Papailiopoulos, Ramchandran, Bartlett). | Yin, Chen, Ramchandran, Bartlett: "Byzantine-robust distributed learning: Towards optimal statistical rates", ICML 2018, PMLR 80, 5650–5659. | PMLR v80 page, arXiv:1803.01498 |
| [17] Raspberry Pi white paper | Year 2020. The document is **Release 1, 1 June 2026** (colophon: build date 12/06/2026). | Raspberry Pi Ltd, "Use-case-specific thermal performance of Raspberry Pi SBCs", white paper, 2026. It supports only: Pi 4 Model B soft throttling at 80 °C, hard throttling at 85 °C on all models, and "minimum power consumption 3–4 W" for the Pi 4 Model B. It gives **no active-load wattage** for the Pi 4, so the benchmark's 5 W figure must be presented as an assumption (the edge model reports it as one). | the PDF itself |

## Minor corrections

| v7 | Change |
|---|---|
| [6] Campos et al. | Title is "…intrusion detection in Internet of Things: Review and challenges" (no "the"). |
| [7] CSE-CIC-IDS2018 | Cite as the owner asks: "A realistic cyber defense dataset (CSE-CIC-IDS2018)", Registry of Open Data on AWS, https://registry.opendata.aws/cse-cic-ids2018 (accessed date). Authors: Canadian Institute for Cybersecurity and Communications Security Establishment. |
| [16] Popoola et al. | Title spells "IoT-Edge devices". Year 2022 is already correct in v7. |
| [11] Kairouz et al. | Title, journal, volume/issue, pages, year and DOI verified (Crossref). The real author list is far longer than ten names, so "et al." after the first ten is acceptable. |

## Verified without change

| v7 | Entry | Checked against |
|---|---|---|
| [1] | Alsaedi et al., IEEE Access 8, 165130–165150 (2020) | Crossref |
| [2] | Bagdasaryan et al., AISTATS 2020, PMLR 108, 2938–2948 | PMLR |
| [3] | Bhagoji et al., ICML 2019, PMLR 97, 634–643 | PMLR |
| [4] | Blanchard et al., NeurIPS 30 (2017), 119–129 | NeurIPS, dblp |
| [8] | Fang et al., USENIX Security 2020, 1605–1622 | USENIX |
| [9] | Garcia, Parmisano, Erquiaga, IoT-23, Zenodo v1.0.0 (2020), DOI 10.5281/zenodo.4743746 | DataCite |
| [10] | Hsu, Qi, Brown, arXiv:1909.06335 (2019) | arXiv |
| [12] | Li et al., MLSys 2020 (vol. 2) | MLSys, dblp (page range 429–450 is the one Google Scholar shows; the sources above do not print it) |
| [13] | McMahan et al., AISTATS 2017, PMLR 54, 1273–1282 | PMLR |
| [14] | Mothukuri et al., FGCS 115, 619–640 (2021) | Crossref |
| [15] | Moustafa & Slay, MilCIS 2015, 1–6 | Crossref |
| [18] | Reddi et al., ICLR 2021 | dblp, arXiv:2003.00295 |
| [19] | Sarhan, Layeghy, Portmann, MONET 27(1), 357–370 (2022) | Crossref |
| [20] | Sharafaldin et al., ICISSP 2018, 108–116 | Crossref |
| [21] | Sun et al., arXiv:1911.07963 | arXiv |
| [22] | Tolpegin et al., ESORICS 2020, 480–501 | Crossref |
| [24] | Zhao et al., arXiv:1806.00582 | arXiv |

## Claims that cite these references

* [17] is cited for the 80 °C soft-throttle point (supported) and for "passive edge micro-computers throttle" (supported).
  It must **not** be cited for an active power of 5 W; that value is an assumption in `src/fedids_bench/evaluation/edge_profile.py`.
* Reviewer 1.8 asks for Chicago or APA reference style. The conference's Springer LNCS template prints numbered references
  (splncs04). Every entry above is complete (authors, title, venue, volume, pages, year, DOI), so either format can be produced from the
  same BibTeX; decide with the conference chair which one applies and say so in the response letter.
