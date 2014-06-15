myfile='ramtest_17-04-2014--22:07:38.mat'
load(myfile)
%%
dd=[];
for i=1:length(d)
try
    dd.doors_to_open{i}= d{i}.doors_to_open;
           dd.xcoord(i)= d{i}.xcoord;
           dd.ycoord(i)= d{i}.ycoord;
     dd.arms_visited{i}= d{i}.arms_visited;
          dd.targets{i}= d{i}.targets;
        dd.diderrors(i)= d{i}.diderrors;
           dd.trials(i)= d{i}.trials;
               dd.ts(i)= d{i}.ts;
            dd.state{i}= d{i}.state;
            dd.didtimeout(i)= d{i}.didtimeout;
          dd.curnarm(i)= d{i}.curnarm;
catch err,err.message

end
end
dd
i
%%
plot(dd.xcoord,dd.ycoord,'.','markersize',1)
try
    gensavefigslabnotebook(['traj' myfile])
end