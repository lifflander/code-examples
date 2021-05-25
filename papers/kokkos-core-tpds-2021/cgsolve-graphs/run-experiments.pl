#!/usr/bin/env perl

use strict;
use warnings;

my @sizes = (16, 32, 64, 128, 256, 512);
my @graph = (0, 1);

print "sizes=@sizes\n";

foreach my $use_graph (@graph) {
    foreach my $n (@sizes) {
        foreach my $i (0..5) {
            #print "N=$n use_graph=$use_graph\n";
            print "jsrun -p 1 ./cgsolve $use_graph $n 2>&1 >run.$i.$use_graph.$n.out\n"
        }
    }
}
