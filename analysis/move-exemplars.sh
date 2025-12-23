#!/bin/bash

while IFS= read -r file; do
    cp "$file" ../exemplars/clean
done < /Users/letitiaho/src/arabic-vowel-task/analysis/best-exemplar-paths.txt
