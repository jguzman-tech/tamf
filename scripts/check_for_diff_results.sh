#!/bin/bash

nm_county_fips=( 011 035 003 059 047 055 017 007 043 006 013 021 023 053 028 033 015 009 041 045 027 019 057 029 031 039 025 005 049 037 001 051 061 )
n_values=( 1 5 10 )

echo "by county results:"
for n in "${n_values[@]}"
do
    count=0
    for fips in "${nm_county_fips[@]}"
    do
	diff -q --from-file ./results/by_county/${n}/${fips}_Utility.*_route_ids.txt > /dev/null
	if [[ "$?" -eq 0 ]]
	then
	    echo "${fips}: Found no differences"
	    count=$((count+1))
	fi
    done
    echo "n=${n}: Found no differences in ${count} out of ${#nm_county_fips[@]}"
    echo ""
done
