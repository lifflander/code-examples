#!/usr/bin/env perl

use strict;
use warnings;

my @sizes = (16, 32, 64, 128, 256);
my @graph = (0, 1);

my $num = 20;

foreach my $use_graph (@graph) {
    open my $out, '>', "graph-2.$use_graph.dat";
    foreach my $n (@sizes) {
        my ($gflops, $nrows, $iter, $time) = (0.0, 0, 0, 0.0);
        foreach my $i (0..$num) {
            my $filename = "output-2/run.$i.$use_graph.$n.out";
            print "Parsing configuration: N=$n use_graph=$use_graph; $filename\n";
            open my $file, '<', $filename;
            for (<$file>) {
                if (/Performance: ([0-9.]+) GFlop/) {
                    # print "gflops=$1\n";
                    if ($1 == 0) {
                        print "failure to parse gflops: $filename\n";
                    }
                    $gflops += $1;
                }
                if (/nrows = ([0-9]+)/) {
                    if ($nrows != 0 && $nrows != $1) {
                        print "inconsistent number of rows: $filename\n";
                    }
                    $nrows = $1;
                    #print "nr=$nrows\n";
                }
                if (/([0-9]+) iterations; ([0-9.]+) time/) {
                    if ($iter != 0 && $iter != $1) {
                        print "inconsistent number of iters: $filename\n";
                    }
                    if ($2 == 0) {
                        print "failure to parse time: $filename\n";
                    }
                    $iter = $1;
                    $time += $2;
                }
            }
            print "$gflops, $nrows, $iter, $time\n";
            close $file;
        }
        my $avgtime = $time/$num;
        my $avggflops = $gflops/$num;
        print $out "$n $nrows $iter $avgtime $avggflops\n";

    }
    close $out;
}
