#!/usr/bin/env perl

use strict;
use warnings;

my @sizes = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576);
my @graph = (0, 1);
my @iters = (10, 50, 100, 200, 500);

print "sizes=@sizes\n";

foreach my $use_graph (@graph) {
    foreach my $n (@sizes) {
	foreach my $it (@iters) {
	    foreach my $i (0..5) {
		print "Running configuration: iters=$it i=$i N=$n use_graph=$use_graph\n";
		print `jsrun -p 1 ./graph_sample $use_graph $n $it 2>&1 >output-1/run.$i.$use_graph.$n.$it.out`;
	    }
	}
    }
}
