function ord = tiedrank(x)
 
 [throwaway,I] = sort(x);
 ord = zeros(size(x));
 
 curVal = x(I(1));
 curSet = zeros(size(x));
 curSetN = 0;
 
 for i=1:length(ord)
 if x(I(i)) == curVal
 curSetN = curSetN+1;
 curSet(curSetN) = i;
 else
 curVal = x(I(i));
 ord(I(curSet(1:curSetN))) = mean(curSet(1:curSetN));
 curSet(1) = i;
 curSetN = 1;
 end
 end
 
 ord(I(curSet(1:curSetN))) = mean(curSet(1:curSetN)); 