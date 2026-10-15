# Review of apparent false positives

Inspected all 15 official-label false positives in the first 200 test pairs.
Judgments use the supplied titles, manufacturers, prices, and positive mappings
across the official train/validation/test files. They are provisional record-level
judgments, not independently verified SKU adjudications.

- Nine look like matches based on the available fields.
- Two look like actual nonmatches involving license/package differences.
- Four remain uncertain because of omitted edition, platform, or bundle details.

Examples of likely matches: Corel VideoStudio 11 Plus (856, 5), SubmitWolf 6.0
with matching SWOLF6R code (427, 2326), Word 2004 Mac upgrade (823, 2474),
and PaperPort Professional 11 five-user license/package (198, 64).

Examples of likely nonmatches: Acrobat Professional 8 Mac Standard-to-Professional
upsell versus unspecified Professional package (508, 533); Creative Suite 3 Design
Standard Mac full package versus nonprofit TLP download media (588, 1842).

The label mappings provide additional evidence of a duplicate-listing problem:
Google record 5 is marked positive with Amazon record 540, titled “ulead
videostudio 11 plus”, but negative with Amazon record 856, “videostudio 11 plus”
by Corel. Elsewhere Google record 2307 is positive with Amazon record 961,
“vegas movie studiodvd +8 plat”, but negative with record 974, “sony vegas movie
studio + dvd 8 platinum edition”. These suggest some negative labels reflect
incomplete or inconsistent match mappings rather than different products.

This sample does not establish how many of the full split's 186 false positives
are mislabeled. It is a review of the first 200 pairs, not a random sample of all
false positives. Original labels, predictions, and official F1 scores are unchanged.
No corrected aggregate F1 is reported from this partial review.

`false_positive_review.csv` contains all 15 pairs, original fields and labels,
model scores, provisional judgments, and reasons.
