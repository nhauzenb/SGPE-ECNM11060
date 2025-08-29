%% y(:,1) = median
%% y(:,2) = 16th percentile
%% y(:,3) = 84th percentile
%% N = Horizon

function plotx2(y,time)
set(gcf,'DefaultAxesColorOrder',[0 0 1;1 0 0;1 0 0;0 0 1]);
c1=y(:,2);
c2=y(:,3);
X1=[(time),fliplr(time)];
X2=[c1',fliplr((c2)')];
hh=fill(X1,X2,'b');
set(hh,'edgecolor',[0.75 0.75 1]);
set(hh,'facecolor',[0.75 0.75 1]);
hold on;
plot(time,y(:,1),'LineWidth',3);
hold on;
plot(time,zeros(length(time),1),':')
end