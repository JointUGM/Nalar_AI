Gold files for the CP draft eval (python -m evals.s1.cp_draft).

Each file is copied BY A HUMAN from an official, public curriculum PDF. Never generate or
"fix" gold text with a model: the eval would then grade the model against itself.

Shape:
  {"decree_code": "046/H/KR/2025" | null,
   "effective_on": "YYYY-MM-DD" | null,
   "statements": ["exact sentence as printed", ...]}

decree_code / effective_on: put the value only if the PDF prints it, otherwise null (the 2022
booklets print no decree number). "statements" are the sentences of the phase outcome
paragraphs, whitespace-normalised, in the PDF's own wording.

Needed (three files):
  cp_ipa_2022.json         "CP IPA Fase D SMP" booklet, kurikulum.kemendikdasmen.go.id
  cp_ipa_bskap_2025.json   the IPA section of BSKAP 046/H/KR/2025
  cp_matematika_d.json     one non-science subject, for example Matematika Fase D

Commit a gold file only when it contains public curriculum text. Never commit the PDFs.
