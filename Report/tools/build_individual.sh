#!/usr/bin/env bash
# Build one copy per student; the declaration names only that student.
# Output: individual/Report_<Name>.pdf
set -e
cd "$(dirname "$0")/.."
mkdir -p individual
while IFS='|' read -r NAME REG FILE; do
  J="Report_${FILE}"
  pdflatex -interaction=nonstopmode -jobname="$J" "\def\soloname{$NAME}\def\soloreg{$REG}\input{thesis}" >/dev/null
  makeindex "$J.nlo" -s nomencl.ist -o "$J.nls" >/dev/null 2>&1 || true
  pdflatex -interaction=nonstopmode -jobname="$J" "\def\soloname{$NAME}\def\soloreg{$REG}\input{thesis}" >/dev/null
  pdflatex -interaction=nonstopmode -jobname="$J" "\def\soloname{$NAME}\def\soloreg{$REG}\input{thesis}" >/dev/null
  mv "$J.pdf" individual/
  rm -f "$J".{aux,log,out,toc,lof,lot,nlo,nls,ilg}
  echo "built individual/$J.pdf"
done <<'LIST'
C S Vaishakh|SJC23AD025|CS_Vaishakh
Elisha Anto|SJC23AD032|Elisha_Anto
Jeffin James|SJC23AD037|Jeffin_James
Rahul Sarang|SJC23AD061|Rahul_Sarang
LIST
