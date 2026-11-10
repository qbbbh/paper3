# All visual outputs in this bundle are generated in R.
suppressPackageStartupMessages({library(ggplot2);library(patchwork);library(svglite)})
args <- commandArgs(trailingOnly=FALSE)
script <- sub('^--file=', '', args[grepl('^--file=',args)])
out <- dirname(normalizePath(script))
d <- read.csv(file.path(out,'source_blocks.csv'),check.names=FALSE,stringsAsFactors=FALSE)
d <- d[tolower(d$paired_ok)=='true',]
d$method <- factor(d$method_plot,levels=c('Reference','No power refinement','Fixed configuration','Proposed'))
pal <- c('Reference'='#7E8994','No power refinement'='#C18A42','Fixed configuration'='#9C8AB8','Proposed'='#247D98')
ltypes <- c('Reference'='dotted','No power refinement'='longdash','Fixed configuration'='longdash','Proposed'='solid')
theme_set(theme_minimal(base_family='Arial',base_size=9)+theme(
  panel.grid.minor=element_blank(),panel.grid.major=element_line(colour='#E8ECF0',linewidth=.25),
  axis.title=element_text(colour='#2F3E49',size=9),axis.text=element_text(colour='#4D5962',size=8),
  plot.title=element_text(face='bold',size=10,colour='#213543',margin=margin(b=5)),
  plot.subtitle=element_text(size=8,colour='#697681',margin=margin(b=9)),
  legend.title=element_blank(),legend.position='top',legend.text=element_text(size=8),
  legend.key.width=grid::unit(1.2,'cm'),plot.margin=margin(8,9,6,8),
  plot.caption=element_text(size=7.4,colour='#667681',hjust=0,margin=margin(t=10)),
  plot.tag=element_text(face='bold',size=11,colour='#213543')))
colors <- function() scale_colour_manual(values=pal,drop=TRUE)
fills <- function() scale_fill_manual(values=pal,drop=TRUE)
savefig <- function(g,name,w=7.4,h=6.4) {
  ggsave(file.path(out,paste0(name,'.png')),g,width=w,height=h,units='in',dpi=320,bg='white',type='cairo')
  ggsave(file.path(out,paste0(name,'.pdf')),g,width=w,height=h,units='in',device=cairo_pdf,bg='white')
  ggsave(file.path(out,paste0(name,'.svg')),g,width=w,height=h,units='in',device=svglite,bg='white')
}
ann <- function(title,sub,caption) plot_annotation(title=title,subtitle=sub,caption=caption,tag_levels='a',theme=theme(
  plot.title=element_text(size=13,face='bold',colour='#213543'),
  plot.subtitle=element_text(size=9,colour='#697681'),
  plot.caption=element_text(size=7.2,colour='#667681',hjust=0),
  plot.margin=margin(10,12,8,8)))

# Figure 1: distributions plus baseline-conditioned, paired benefit.
long <- d[d$cohort %in% c('10_long_A','10_long_B'),]
qstats <- list(); ptop <- list(); pbottom <- list()
for(i in 1:2) {
  cohort <- c('10_long_A','10_long_B')[i]
  z <- long[long$cohort==cohort,]
  ref <- z[z$method=='Reference',]
  ref <- ref[order(ref$ref_min,ref$block),]
  ref$quartile <- pmin(4,ceiling(seq_len(nrow(ref))*4/nrow(ref)))
  z$quartile <- ref$quartile[match(z$block,ref$block)]
  st <- do.call(rbind,lapply(split(z[z$method!='Reference',],list(z$method[z$method!='Reference'],z$quartile[z$method!='Reference']),drop=TRUE),function(v) {
    data.frame(cohort=cohort,method=as.character(v$method[1]),quartile=v$quartile[1],n=nrow(v),
               med=median(v$gain),lo=as.numeric(quantile(v$gain,.25)),hi=as.numeric(quantile(v$gain,.75)),
               improved=mean(v$gain>1e-9)*100,ref_lo=min(v$ref_min),ref_hi=max(v$ref_min))
  }))
  st$method <- factor(st$method,levels=levels(d$method)); qstats[[i]] <- st
  z$method <- droplevels(z$method)
  ptop[[i]] <- ggplot(z,aes(min_ee,colour=method,linetype=method))+stat_ecdf(geom='step',linewidth=.85,pad=FALSE)+
    colors()+scale_linetype_manual(values=ltypes)+
    scale_y_continuous(labels=scales::label_percent(accuracy=1),breaks=c(0,.25,.5,.75,1),limits=c(0,1.02))+ 
    scale_x_continuous(breaks=seq(0,400,100),limits=c(0,400))+
    labs(title=paste0('Channel realization ',c('A','B')[i]),
         subtitle=paste0(nrow(ref),' paired blocks | farther right is better'),
         x='Actual weakest-link EE (Mbit/J)',y='Cumulative share of blocks')
  pbottom[[i]] <- ggplot(st,aes(quartile,med,colour=method,group=method))+
    geom_hline(yintercept=0,colour='#7E8994',linewidth=.35)+
    geom_linerange(aes(ymin=lo,ymax=hi),position=position_dodge(width=.25),linewidth=1.2,alpha=.65)+
    geom_line(aes(linetype=method),position=position_dodge(width=.25),linewidth=.65)+
    geom_point(aes(shape=method),position=position_dodge(width=.25),size=2.6,fill='white',stroke=.9)+
    colors()+scale_linetype_manual(values=ltypes)+scale_shape_manual(values=c('Proposed'=21,'No power refinement'=23))+
    scale_x_continuous(breaks=1:4,labels=c('Q1\nWeakest','Q2','Q3','Q4\nStrongest'))+
    scale_y_continuous(limits=c(-10,100),breaks=c(0,25,50,75,100))+
    labs(title='Benefit across baseline conditions',subtitle='Paired gain: median and interquartile range',
         x='Quartile of reference weakest-link EE',y='Weakest-link EE gain (Mbit/J)')+theme(legend.position='none')
}
write.csv(do.call(rbind,qstats),file.path(out,'quartile_response.csv'),row.names=FALSE)
f1 <- ((ptop[[1]]|ptop[[2]])/(pbottom[[1]]|pbottom[[2]]))+plot_layout(guides='collect',heights=c(1,1))+
  ann('Continuous refinement: benefit distribution',
      '10 vehicles | same traffic, two channel realizations | 99.9-s window | prediction budget = 10%',
      'Each observation is one block. Quartiles are formed separately within each realization using the reference.\nIntervals describe block variability, not confidence intervals. Realization B has one jointly failed block, excluded from paired curves.')
savefig(f1,'fig01_distribution_and_weak_blocks',h=6.8)

# Figure 2: full paired trade-off cloud, including tails and unfavorable outcomes.
# The x-axis uses a declared signed asinh transformation so that all observations remain visible.
trade <- list(); regions <- list()
for(i in 1:2) for(j in 1:2) {
  cohort <- c('10_long_A','10_long_B')[i]; meth <- c('No power refinement','Proposed')[j]
  z <- long[long$cohort==cohort & long$method==meth,]
  n <- nrow(z); positive <- mean(z$gain>1e-9)*100
  inside <- mean(z$gain>1e-9 & z$loss_pct<=10+1e-8)*100
  worse <- mean(z$gain < -1e-9)*100
  exceed <- mean(z$loss_pct>10+1e-8)*100
  regions[[length(regions)+1]] <- data.frame(cohort=cohort,method=meth,n=n,improved_pct=positive,
    improved_within_actual_10pct=inside,worse_pct=worse,actual_loss_over_10pct=exceed,
    unchanged_pct=mean(abs(z$gain)<=1e-9)*100)
  trade[[length(trade)+1]] <- ggplot(z,aes(loss_pct,gain))+
    annotate('rect',xmin=-200,xmax=10,ymin=0,ymax=280,fill='#EEF5F5',alpha=.8)+
    geom_hline(yintercept=0,colour='#7E8994',linewidth=.45)+
    geom_vline(xintercept=10,colour='#A37A72',linetype='dashed',linewidth=.55)+
    geom_point(colour=pal[meth],alpha=.25,size=1.2,stroke=0)+
    geom_point(data=z[tolower(z$qos_ok)=='false',],shape=4,size=2,stroke=.65,colour='#263744')+
    geom_point(aes(x=0,y=0),data=data.frame(loss_pct=0,gain=0),shape=21,size=3,fill='white',colour='#667681',stroke=.8)+
    scale_x_continuous(trans=scales::pseudo_log_trans(sigma=3),breaks=c(-150,-50,-10,0,5,10,25),limits=c(-200,40))+
    scale_y_continuous(breaks=c(-80,0,80,160,240),limits=c(-100,280))+
    labs(title=paste0(c('A','B')[i],'  |  ',meth),
         subtitle=sprintf('Improved: %.1f%%\nGain with actual loss <=10%%: %.1f%%',positive,inside),
         x='Actual mean-EE loss (%) [asinh scale]',y='Weakest-link EE gain (Mbit/J)')+
    annotate('text',x=-190,y=265,label=sprintf('n = %d\nMedian gain: %.1f Mbit/J\nLoss >10%%: %.1f%%',n,median(z$gain),exceed),
             hjust=0,vjust=1,size=2.65,colour='#445661',lineheight=1.25)+
    annotate('text',x=10,y=-93,label='10%',size=2.6,hjust=-.2,colour='#946E67')
}
write.csv(do.call(rbind,regions),file.path(out,'tradeoff_regions.csv'),row.names=FALSE)
f2 <- wrap_plots(trade,ncol=2)+ann('What does each block gain, and what does it cost?',
   '10 vehicles | proposed versus no continuous refinement | changes relative to the same-block mean-EE reference',
   'Every point is a paired block; no tail observations are discarded. Pale region: positive minimum-EE gain and actual loss <=10%.\nDashed line marks 10% actual loss for evaluation, not an execution guarantee. Crosses: actual block QoS failures; open circle: reference.\nSigned asinh x-axis preserves the sign and displays the full loss/gain range; displayed tick labels are actual percentages.')
savefig(f2,'fig02_paired_tradeoff',h=6.7)

# Figure 3: fixed-configuration comparisons, kept in their own matched batches.
cohorts <- c('10_short_A','50_short_A','50_short_B')
heads <- c('10 vehicles | A','50 vehicles | A','50 vehicles | B')
same_stats <- list(); same_blocks <- list(); top <- list(); bot <- list()
for(i in 1:3) {
  z <- d[d$cohort==cohorts[i],]; z$method <- droplevels(z$method)
  a <- z[z$method=='Proposed',]; b <- z[z$method=='Fixed configuration',]
  a <- a[order(a$block),]; b <- b[match(a$block,b$block),]
  dd <- data.frame(cohort=cohorts[i],block=a$block,t=a$start_time_s,ref_min=a$ref_min,
                  full=a$min_ee,fixed=b$min_ee,difference=a$min_ee-b$min_ee,
                  mean_difference=a$mean_ee-b$mean_ee)
  eq <- abs(dd$difference)<=1e-9; n <- nrow(dd)
  same_stats[[i]] <- data.frame(cohort=cohorts[i],n=n,equal=sum(eq),higher=sum(dd$difference>1e-9),
                               lower=sum(dd$difference< -1e-9),mean_diff=mean(dd$difference),median_diff=median(dd$difference))
  same_blocks[[i]] <- dd
  top[[i]] <- ggplot(z,aes(min_ee,colour=method,linetype=method))+stat_ecdf(geom='step',linewidth=.85,pad=FALSE)+
    colors()+scale_linetype_manual(values=ltypes)+
    scale_y_continuous(breaks=c(0,.5,1),labels=scales::label_percent(),limits=c(0,1.03))+
    scale_x_continuous(limits=c(0,if(i==1)340 else 120),breaks=if(i==1)c(0,100,200,300) else c(0,40,80,120))+
    labs(title=heads[i],subtitle='100 paired blocks; 10-s window',x='Weakest-link EE (Mbit/J)',y=if(i==1)'Cumulative share' else NULL)
  bot[[i]] <- ggplot(dd,aes(t,difference))+
    geom_hline(yintercept=0,linewidth=.5,colour='#7E8994')+
    geom_segment(aes(xend=t,yend=0),colour=pal['Fixed configuration'],linewidth=.4,alpha=.8)+
    geom_point(shape=21,fill=pal['Proposed'],colour='white',size=1.7,stroke=.2)+
    scale_y_continuous(limits=c(-22,22),breaks=c(-20,-10,0,10,20))+
    labs(title=paste0(sum(eq),' / ',n,' blocks equal'),
         subtitle=sprintf('Higher %d  |  lower %d',sum(dd$difference>1e-9),sum(dd$difference< -1e-9)),
         x='Block start time (s)',y=if(i==1)'Full - fixed configuration\nweakest-link EE (Mbit/J)' else NULL)+
    annotate('text',x=min(dd$t),y=21,label=sprintf('Mean difference\n%+.3f Mbit/J',mean(dd$difference)),
             hjust=0,vjust=1,size=2.8,colour='#445661',lineheight=1.2)
}
write.csv(do.call(rbind,same_stats),file.path(out,'fixed_configuration_summary.csv'),row.names=FALSE)
write.csv(do.call(rbind,same_blocks),file.path(out,'fixed_configuration_block_differences.csv'),row.names=FALSE)
f3 <- (wrap_plots(top,nrow=1)/wrap_plots(bot,nrow=1))+plot_layout(guides='collect')+
  ann('How much does reconfiguring the links add?',
      'Matched short-window experiments | same configuration retains continuous power optimization',
      'Top: full and fixed-configuration distributions largely overlap. Bottom: all individual differences, including negative values.\nEquality tolerance: 1e-9 Mbit/J. This 10-vehicle batch uses a different evaluation window and source snapshot from Figures 1-2; kept separate.\nNo 50-vehicle no-refinement batch is available. A and B denote channel realizations, not independent traffic scenes.')
savefig(f3,'fig03_fixed_configuration',h=6.0)

# A compact block-outcome map is an additional explanation view, not a second performance ranking.
long2 <- long[long$method!='Reference',]
long2$seed <- ifelse(long2$cohort=='10_long_A','Channel realization A','Channel realization B')
long2$outcome <- factor(ifelse(long2$gain>1e-9,'Higher',ifelse(long2$gain< -1e-9,'Lower','Unchanged')),
                       levels=c('Higher','Unchanged','Lower'))
cnt <- aggregate(block~seed+method+outcome,long2,length)
names(cnt)[names(cnt)=='block'] <- 'count'
tot <- aggregate(count~seed+method,cnt,sum);names(tot)[3] <- 'total'
cnt <- merge(cnt,tot,by=c('seed','method')); cnt$pct <- cnt$count/cnt$total
cnt$label <- sprintf('%.1f%%\n%d blocks',100*cnt$pct,cnt$count)
covplot <- ggplot(cnt,aes(outcome,method,fill=outcome))+
  geom_tile(aes(alpha=.3+.7*pct),width=.94,height=.86,colour='white',linewidth=.7)+
  geom_text(aes(label=label),size=3.4,colour='#213543',lineheight=1.25)+
  facet_wrap(~seed,ncol=2)+scale_alpha_identity()+
  scale_fill_manual(values=c(Higher='#93C4CD',Unchanged='#DDE2E6',Lower='#DDB8A7'))+
  scale_y_discrete(labels=c('Proposed'='Proposed','No power refinement'='No continuous\nrefinement'))+
  labs(x='Change in actual weakest-link EE',y=NULL,title='How often does weakest-link EE improve?',
       subtitle='Outcome counts relative to the same-block mean-EE reference',
       caption='All paired blocks are classified, including unchanged and unfavorable outcomes.\nA: 999 paired blocks. B: 998 paired blocks; one failed block excluded. Equality tolerance: 1e-9 Mbit/J.')+
  theme(legend.position='none',strip.text=element_text(face='bold',colour='#213543'),panel.grid=element_blank(),
        panel.spacing=grid::unit(.7,'cm'),plot.title.position='plot',plot.caption.position='plot')
savefig(covplot,'fig04_outcome_coverage',h=3.3,w=7.4)
capture.output(sessionInfo(),file=file.path(out,'R_sessionInfo.txt'))
cat('Created four figure sets (PNG / PDF / SVG).\n')
print(do.call(rbind,qstats));print(do.call(rbind,regions));print(do.call(rbind,same_stats))
