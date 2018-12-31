# Compare your discrete ROC curves with other methods
# At terminal: gnuplot contROC.p
set terminal png size 1280, 960 enhanced font 'Verdana,18'
set key font ",12"
set size 1,1
set xtics 50
set ytics 0.1
set grid
set ylabel "True positive rate"
set xlabel "False positive"
set xr [0:400]
set yr [0:1.0]
# Compare your discrete ROC curves with other methods
# At terminal: gnuplot discROC.p
set key below
set output "contROC-compare.png"
plot  "tempContROC.txt" using 2:1 title 'Ours(PRCO)' with linespoints pointinterval 80 lw 3 ,\
"rocCurves/DDFD_ContROC.txt" using 2:1 title 'DDFD' with lines lw 2 , \
"rocCurves/CasCNN-ContROC.txt" using 2:1 title 'CascadeCNN' with lines lw 2 , \
"rocCurves/jjyan_allROC_ContROC.txt" using 2:1 title 'Yan et al.' with lines lw 2 , \
"rocCurves/AcfContROC.txt" using 2:1 title 'ACF-multiscale' with lines lw 2 , \
"rocCurves/pico-ContROC.txt" using 2:1 title 'Pico' with lines lw 2 , \
"rocCurves/HeadHunter_ContROC.txt" using 2:1 title 'HeadHunter' with lines lw 2 , \
"rocCurves/BoostedExamplerBased-ContROC.txt" using 2:1 title 'Boosted Exemplar' with lines lw 2 , \
"rocCurves/SURF_GentleBoost_FTContROC.txt" using 2:1 title 'SURF-frontal' with lines lw 2 , \
 "rocCurves/SURF_GentleBoost_MVContROC.txt" using 2:1 title 'SURF-multiview' with lines lw 2 , \
 "rocCurves/PEPAdapt_ContROC.txt" using 2:1 title 'PEP-Adapt' with lines lw 2 , \
 "rocCurves/XZJY_ContROC.txt" using 2:1 title 'XZJY' with lines lw 2 


 
