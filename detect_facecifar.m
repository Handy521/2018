function [total_boxes points] = detect_face(img,minsize,PNet,claNet,RNet,ONet,threshold,fastresize,factor)
	%im: input image
	%minsize: minimum of faces' size
	%pnet, rnet, onet: caffemodel
	%threshold: threshold=[th1 th2 th3], th1-3 are three steps's threshold
	%fastresize: resize img from last scale (using in high-resolution images) if fastresize==true
	factor_count=0;
	total_boxes=[];
	points=[];
	h=size(img,1);
	w=size(img,2);    
	minl=min([w h]);
    img=single(img);
    if fastresize
        im_data=(single(img)-127.5)*0.0078125;
    end
    m=12/minsize;
	minl=minl*m;
	%creat scale pyramid
    scales=[];
	while (minl>=12)
		scales=[scales m*factor^(factor_count)];
		minl=minl*factor;
		factor_count=factor_count+1;
	end
	%first stage
    for j = 1:size(scales,2)
        scale=scales(j);
        hs=ceil(h*scale);
        ws=ceil(w*scale);
        if fastresize
            im_data=imResample(im_data,[hs ws],'bilinear');
        else
            im_data=(imResample(img,[hs ws],'bilinear')-127.5)*0.0078125;
        end
        PNet.blobs('data').reshape([hs ws 3 1]);
        out=PNet.forward({im_data});
        boxes=generateBoundingBox(out{2}(:,:,2),out{1},scale,threshold(1));
        %inter-scale nms
        pick=nms(boxes,0.5,'Union');
        boxes=boxes(pick,:);
        if ~isempty(boxes)
            total_boxes=[total_boxes;boxes];
        end
    end
    if ~isempty(total_boxes)
        pick=nms(total_boxes,0.7,'Union');
        total_boxes=total_boxes(pick,:);
        regw=total_boxes(:,3)-total_boxes(:,1);
        regh=total_boxes(:,4)-total_boxes(:,2);
        total_boxes=[total_boxes(:,1)+total_boxes(:,6).*regw total_boxes(:,2)+total_boxes(:,7).*regh total_boxes(:,3)+total_boxes(:,8).*regw total_boxes(:,4)+total_boxes(:,9).*regh total_boxes(:,5)];
        total_boxes=rerec(total_boxes);
        total_boxes(:,1:4)=fix(total_boxes(:,1:4));
        [dy edy dx edx y ey x ex tmpw tmph]=pad(total_boxes,w,h);
    end
    numbox=size(total_boxes,1);
    
    if numbox>0
        %second stage   %对相应的框、回归量和点进行分类
        tempimg=zeros(32,32,3,numbox);
        tempimg24=zeros(24,24,3,numbox);
        for k=1:numbox
            tmp=zeros(tmph(k),tmpw(k),3);
            tmp(dy(k):edy(k),dx(k):edx(k),:)=img(y(k):ey(k),x(k):ex(k),:);
            tempimg24(:,:,:,k)=imResample(tmp,[24 24],'bilinear');
            tmp=tmp(:,:,[3,2,1]);%RGB to BGR
            tmp=permute(tmp,[2,1,3]);
            tempimg(:,:,:,k)=imResample(tmp,[32 32],'bilinear');
        end
        claNet.blobs('data').reshape([32 32 3 numbox]);
        claNet.forward({tempimg});
        feat=claNet.blobs('prob').get_data();
        [xx,b]=max(feat);
        total_boxes=[total_boxes,b'];
        rotate=find(total_boxes(:,6)>1&total_boxes(:,5)>0.60); %降低这个值检测到所有框
        save_rotate=total_boxes(rotate,:);%save rotate box
        %3 stage
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        tempimg24=(tempimg24-127.5)*0.0078125;
        RNet.blobs('data').reshape([24 24 3 numbox]);%1 stage output
        out=RNet.forward({tempimg24});
        score=squeeze(out{2}(2,:));
        pass=find(score>threshold(2));
        total_boxes=[total_boxes(pass,1:4) score(pass)'];
        mv=out{1}(:,pass);
        if size(total_boxes,1)>0
            pick=nms(total_boxes,0.7,'Union');
            total_boxes=total_boxes(pick,:);
            total_boxes=bbreg(total_boxes,mv(:,pick)');
            total_boxes=rerec(total_boxes);
        end
        numbox2=size(total_boxes,1);
        if numbox2>0
            %4 stage
            total_boxes=fix(total_boxes);
            [dy edy dx edx y ey x ex tmpw tmph]=pad(total_boxes,w,h);
            tempimg=zeros(48,48,3,numbox2);
            for k=1:numbox2
                tmp=zeros(tmph(k),tmpw(k),3);
                tmp(dy(k):edy(k),dx(k):edx(k),:)=img(y(k):ey(k),x(k):ex(k),:);
                tempimg(:,:,:,k)=imResample(tmp,[48 48],'bilinear');
            end
            tempimg=(tempimg-127.5)*0.0078125;
            ONet.blobs('data').reshape([48 48 3 numbox2]);
            out=ONet.forward({tempimg});
            score=squeeze(out{3}(2,:));
            points=out{2};
            pass=find(score>threshold(3));
            points=points(:,pass);
            total_boxes=[total_boxes(pass,1:4) score(pass)'];
            amv=out{1}(:,pass);%save reg          
            w1=total_boxes(:,3)-total_boxes(:,1)+1;
            h1=total_boxes(:,4)-total_boxes(:,2)+1;
            points(1:5,:)=repmat(w1',[5 1]).*points(1:5,:)+repmat(total_boxes(:,1)',[5 1])-1;%first appear points
            points(6:10,:)=repmat(h1',[5 1]).*points(6:10,:)+repmat(total_boxes(:,2)',[5 1])-1;
            apoints=points;%save points
            if size(total_boxes,1)>0
                total_boxes=bbreg(total_boxes,amv(:,:)');
            end
        end
        % 5 stage
        clanumbox=size( save_rotate,1);rbox=[];rpoints=[];
        if clanumbox>0
            [rdy redy rdx redx ry rey rx rex rtmpw rtmph]=pad(save_rotate,w,h);
            rtempimg=zeros(48,48,3,clanumbox);
            for k=1:clanumbox
                rtmp=zeros(rtmph(k),rtmpw(k),3);
                rtmp(rdy(k):redy(k),rdx(k):redx(k),:)=img(ry(k):rey(k),rx(k):rex(k),:);
                switch save_rotate(k,6)
                    case {2}
                        rtmp=imrotate(rtmp,90);
                    case{3}
                        rtmp=imrotate(rtmp,180);
                    otherwise
                        rtmp=imrotate(rtmp,270);
                end
                rtempimg(:,:,:,k)=imResample(rtmp,[48 48],'bilinear');
            end
            rtempimg=(rtempimg-127.5)*0.0078125;
            ONet.blobs('data').reshape([48 48 3 clanumbox]);
            out=ONet.forward({rtempimg});
            score=squeeze(out{3}(2,:));
            points=out{2};
            pass=find(score>threshold(4));
            points=points(:,pass);
            bboxes=[save_rotate(pass,1:4) score(pass)' save_rotate(pass,6)];%rotate box
            mv=out{1}(:,pass);
            if size(bboxes,1)>0
                
                box=[];cox=[];ddox=[];bbpoints=[];ccpoints=[];ddpoints=[];
                bbmv=[];ccmv=[];ddmv=[];
                for i=1:size(bboxes)
                    switch bboxes(i,6)%对相应的框、回归量和点进行分类
                        case {2}
                            bb=bboxes(i,:);
                            box=[box;bb];
                            bpoints=points(:,i);
                            bbpoints=[bbpoints,bpoints];
                            bmv=mv(:,i);
                            bbmv=[bbmv,bmv];
                        case{3}
                            cc=bboxes(i,:);
                            cox=[cox;cc];
                            cpoints=points(:,i);
                            ccpoints=[ccpoints,cpoints];
                            cmv=mv(:,i);
                            ccmv=[ccmv,cmv];
                        otherwise
                            dd1=bboxes(i,:);
                            ddox=[ddox;dd1];
                            dpoints=points(:,i);
                            ddpoints=[ddpoints,dpoints];
                            dmv=mv(:,i);
                            ddmv=[ddmv,dmv];
                    end
                end
                if ~isempty(box)%rotate 90
                    numP=size(bbpoints,2);
                    c=reshape(bbpoints,1,10*numP);
                    jj=0;
                    for i=0:5:5*numP-1
                        dd(1,:)=c(1+2*i:5+2*i);
                        dd(2,:)=c(6+2*i:10+2*i);
                        dd(3,:)=1;
                        M=[1   0    0.5;
                            0   1   0.5;
                            0    0    1  ];
                        M1=[cos(pi/2)   sin(pi/2)   0;%90
                            -sin(pi/2)   cos(pi/2)  0;
                            0           0         1];
                        M2=[1    0    -0.5;
                            0    1    -0.5;
                            0    0       1  ];
                        mm=M*M1*M2;
                        ccc(:,1+i:5+i)  =mm*dd;
                        jj=jj+1;
                        bbpoints(1:5,jj)=ccc(1,1+i:5+i);
                        bbpoints(6:10,jj)=ccc(2,1+i:5+i);
                    end
                    %得到点和回归后的框
                    w1=box(:,3)-box(:,1)+1;
                    h1=box(:,4)-box(:,2)+1;
                    bbpoints(1:5,:)=repmat(box(:,3)',[5 1])-repmat(w1',[5 1]).*bbpoints(1:5,:);
                    bbpoints(6:10,:)=repmat(box(:,4)',[5 1])-repmat(h1',[5 1]).*bbpoints(6:10,:);
                    box=bbre901(box,bbmv(:,:)');
                end
                if ~isempty(cox)%rotate180
                    w1=cox(:,3)-cox(:,1)+1;
                    h1=cox(:,4)-cox(:,2)+1;
                    ccpoints(6:10,:)=repmat(cox(:,4)',[5 1])-repmat(h1',[5 1]).*ccpoints(6:10,:)+1;
                    ccpoints(1:5,:)=repmat(cox(:,3)',[5 1])-repmat(w1',[5 1]).*ccpoints(1:5,:)+1;
                    cox=bbreg2(cox,ccmv(:,:)');
                end
                if ~isempty(ddox)%rotate270
                    numP=size(ddpoints,2);
                    c=reshape(ddpoints,1,10*numP);
                    jj=0;
                    for i=0:5:5*numP-1
                        dd(1,:)=c(1+2*i:5+2*i);
                        dd(2,:)=c(6+2*i:10+2*i);
                        dd(3,:)=1;
                        M=[1   0    0.5;
                            0   1   0.5;
                            0    0    1  ];
                        M1=[cos(3*pi/2)   sin(3*pi/2)   0;
                            -sin(3*pi/2)   cos(3*pi/2)  0;
                            0           0         1];
                        M2=[1    0    -0.5;
                            0    1    -0.5;
                            0    0       1  ];
                        mm=M*M1*M2;
                        ddd(:,1+i:5+i)  =mm*dd;
                        jj=jj+1;
                        ddpoints(1:5,jj)=ddd(1,1+i:5+i);
                        ddpoints(6:10,jj)=ddd(2,1+i:5+i);
                    end
                    w1=ddox(:,3)-ddox(:,1)+1;
                    h1=ddox(:,4)-ddox(:,2)+1;
                    ddpoints(1:5,:)=repmat(ddox(:,3)',[5 1])-repmat(w1',[5 1]).*ddpoints(1:5,:);
                    ddpoints(6:10,:)=repmat(ddox(:,4)',[5 1])-repmat(h1',[5 1]).*ddpoints(6:10,:);
                    ddox=bbre270(ddox,ddmv(:,:)');
                end
                rbox=[box;cox;ddox];
                %rbox= rbox(:,1:5);
                rpoints=[bbpoints,ccpoints,ddpoints];
                pick=nms(rbox,0.44,'Min');%former 0.7
                rbox=rbox(pick,1:5);
                rpoints=rpoints(:,pick);
            end
        end%旋转框处理完
        total_boxes=[total_boxes;rbox];
        apoints=[apoints,rpoints];
        pick=nms(total_boxes,0.7,'Min');
        total_boxes=total_boxes(pick,:);
        points=apoints(:,pick);
    end
end

