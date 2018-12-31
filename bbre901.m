function [boundingbox] = bbre901(boundingbox,reg)
	%calibrate bouding boxes
	if size(reg,2)==1
		reg=reshape(reg,[size(reg,3) size(reg,4)])';
	end
	w1=[boundingbox(:,3)-boundingbox(:,1)]+1;
	h1=[boundingbox(:,4)-boundingbox(:,2)]+1;
    boundingbox(:,1:4)=[boundingbox(:,1)-reg(:,4).*w1 boundingbox(:,2)+reg(:,1).*h1 boundingbox(:,3)-reg(:,2).*w1 boundingbox(:,4)+reg(:,3).*h1];
    %boundingbox(:,1:4)=[boundingbox(:,1)+reg(:,1).*w1 boundingbox(:,2)+reg(:,2).*h1 boundingbox(:,3)+reg(:,3).*w1 boundingbox(:,4)-reg(:,4).*h1];
     %boundingbox(:,1:4)=[boundingbox(:,1)+reg(:,1).*w1 boundingbox(:,2)+reg(:,1).*h1 boundingbox(:,3)-reg(:,3).*w1 boundingbox(:,4)-reg(:,4).*h1];
   % boundingbox(:,1:4)=[boundingbox(:,1)+reg(:,4).*w1 boundingbox(:,2)+reg(:,1).*h1 boundingbox(:,3)+reg(:,2).*w1 boundingbox(:,4)+reg(:,3).*h1];
    %boundingbox(:,1:4)=[boundingbox(:,1)-reg(:,3).*w1 boundingbox(:,2)+reg(:,2).*h1 boundingbox(:,3)+reg(:,1).*w1 boundingbox(:,4)-reg(:,4).*h1];
    %boundingbox(:,1:4)=[boundingbox(:,1)-reg(:,3).*w1 boundingbox(:,2)-reg(:,2).*h1 boundingbox(:,3)+reg(:,1).*w1 boundingbox(:,4)+reg(:,4).*h1];
   % boundingbox(:,1:4)=[boundingbox(:,1)+reg(:,3).*w1 boundingbox(:,2)+reg(:,2).*h1 boundingbox(:,3)+reg(:,1).*w1 boundingbox(:,4)+reg(:,4).*h1];
    %boundingbox(:,1:4)=[h-boundingbox(:,4)+reg(:,1).*w1 boundingbox(:,1)+reg(:,2).*h1 h-boundingbox(:,2)+reg(:,3).*w1 boundingbox(:,3)+reg(:,4).*h1];
	%boundingbox(:,1:4)=[h-boundingbox(:,4)+reg(:,4).*w1 boundingbox(:,1)+reg(:,1).*h1 h-boundingbox(:,2)+reg(:,2).*w1 boundingbox(:,3)+reg(:,3).*h1];
    %boundingbox(:,1:4)=[boundingbox(:,2)+reg(:,2).*w1 w-boundingbox(:,3)+reg(:,3).*h1 boundingbox(:,4)+reg(:,4).*w1 w-boundingbox(:,1)+reg(:,1).*h1];

end

