#!/usr/bin/env perl

use strict;
use warnings;

my $output = $ARGV[0];

print "output directory=$output\n";

my @sizes = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304);
my @graph = (0, 1);
my @iters = (10, 20, 30, 40, 50, 100, 200, 500);

my $num = 5;

sub average{
    my($data) = @_;
    if (not @$data) {
        die("Empty arrayn");
    }
    my $total = 0;
    foreach (@$data) {
        $total += $_;
    }
    my $average = $total / @$data;
    return $average;
}
sub stdev {
    my($data) = @_;
    if(@$data == 1){
        return 0;
    }
    my $average = &average($data);
    my $sqtotal = 0;
    foreach(@$data) {
        $sqtotal += ($average-$_) ** 2;
    }
    my $std = ($sqtotal / (@$data-1)) ** 0.5;
    return $std;
}

foreach my $use_graph (@graph) {
    open my $out, '>', "graph.$use_graph.dat";
    foreach my $n (@sizes) {
	foreach my $it (@iters) {
	    my (@full_time, @part_time);
	    foreach my $i (0..$num) {
		my $filename = "$output/run.$i.$use_graph.$n.$it.out";
		print "Parsing configuration: N=$n iter=$it use_graph=$use_graph; $filename\n";
		open my $file, '<', $filename;
		for (<$file>) {
		    if (/full_time=([0-9.]+), part_time=([0-9.]+)/) {
			if ($1 == 0 or $2 == 0) {
			    print "failure to parse time: $filename\n";
			}
			push @full_time, ($1/$it)*1000*1000;
			push @part_time, ($2/($it-1))*1000*1000;
		    }
		}
		close $file;
	    }
	    my $avgftime = &average(\@full_time);
	    my $stdftime = &stdev(\@full_time);
	    my $avgptime = &average(\@part_time);
	    my $stdptime = &stdev(\@part_time);
	    print $out "$n $it $avgftime $stdftime $avgptime $stdptime\n";
	}
    }
    close $out;
}
