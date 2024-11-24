## 2.🔎SSD
**들어가기 전 우선 이 장에서 저자가 코드가 매우 어렵기에 코드를 하나하나 다 이해하는 것보다 흐름을 보라고 권장한다.**
### ✏️2.5 순전파 함수 구현
- 물체 감지의 경우 더 복잡한 순전파 함수 필요
-> **Non-Maximum Suppression** 필요

#### 💬 2.5.1 decode 함수 구현
- decode 함수와 nm_suppression 함수를 구현해보자

**<deocode 함수>**
- DBox 정보와 SSD 모델에서 구한 오프셋 정보 `loc`을 사용하여 BBOX 좌표 정보를 생성한다

- 식을 이용하여 BBOX좌표 정보를 (xmin,ymin,xmax,ymax)로 변환한다
```
# 오프셋 정보로 DBOX를 BBOX로 변환하는 함수


def decode(loc, dbox_list):
    """
    오프셋 정보로 DBOX를 BBOX로 변환한다.

    Parameters
    ----------
    loc:  [8732,4]
        SSD모델로 추론하는 오프셋 정보
    dbox_list: [8732,4]
        DBox 정보

    Returns
    -------
    boxes : [xmin, ymin, xmax, ymax]
        BBOX 정보
    """

    # DBox는[cx, cy, width, height]로 저장되어 있다
    # loc는[Δcx, Δcy, Δwidth, Δheight]로 저장되어 있다

    # 오프셋 정보로  BBOX를 구한다
    boxes = torch.cat(( # BBOX 계산하는 공식 구현
        dbox_list[:, :2] + loc[:, :2] * 0.1 * dbox_list[:, 2:],
        dbox_list[:, 2:] * torch.exp(loc[:, 2:] * 0.2)), dim=1)
    # boxes의 크기는 torch.Size([8732, 4])가 된다

    # BBox의 좌표 정보를 [cx, cy, width, height]에서[xmin, ymin, xmax, ymax] 로 변경
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]

    return boxes

```
#### 💬 2.5.2 Non_Maximum Suppression 함수 구현
- 화상 속 동일한 물체에 다른 BBOX가 조금 다르게 복수 피팅하는 경우를 방지하기 위해 겹치는 BBOX를 제거하고 하나의 물체에 하나의 BBOX만 남기는 작업
-> 임계값 `conf`를 지정하여 임계값 이상의 BBOX들을 복수 피팅되었다고 가정하고 그 중 신뢰도 `conf`가 가장 높은 BBOX만 남기고 나머지 삭제
```
# Non-Maximum Suppression을 수행하는 함수
def nm_suppression(boxes, scores, overlap=0.45, top_k=200):
    """
    Non-Maximum Suppression을 수행하는 함수.
    boxes 중에서 너무 많이 겹치는(overlap 이상) BBox를 제거한다.

    Parameters
    ----------
    boxes : [신뢰도 임계값(0.01)을 초과한 BBox 수,4]
        BBox 정보.
    scores : [신뢰도 임계값(0.01)을 초과한 BBox 수]
        conf의 정보

    Returns
    -------
    keep : 리스트
        conf의 내림차순으로 nms를 통과한 index가 저장
    count : int
        nms를 통과한 BBox의 수
    """

    # return의 초기 형태를 생성
    count = 0
    keep = scores.new(scores.size(0)).zero_().long()
    # keep: torch.Size([신뢰도 임계값을 초과한 BBox 수]), 요소는 모두 0

    # 각 BBox의 면적(area)을 계산
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)

    # boxes를 복사한다. 나중에 BBox의 겹침 정도(IOU) 계산에 사용할 템플릿으로 준비
    tmp_x1 = boxes.new()
    tmp_y1 = boxes.new()
    tmp_x2 = boxes.new()
    tmp_y2 = boxes.new()
    tmp_w = boxes.new()
    tmp_h = boxes.new()

    # score를 오름차순으로 정렬
    v, idx = scores.sort(0)

    # 상위 top_k개(200개)의 BBox의 index를 추출 (200개가 존재하지 않을 수도 있음)
    idx = idx[-top_k:]

    # idx의 요소 수가 0이 아닌 동안 반복
    while idx.numel() > 0:
        i = idx[-1]  # 현재 conf 최대의 index를 i에

        # keep의 현재 마지막에 conf 최대의 index를 저장
        # 이 index의 BBox와 많이 겹치는 BBox를 이후 삭제할 예정
        keep[count] = i
        count += 1

        # 마지막 BBox인 경우 루프를 종료
        if idx.size(0) == 1:
            break

        # 현재 conf 최대의 index를 keep에 저장했으므로, idx를 하나 줄인다
        idx = idx[:-1]

        # -------------------
        # 이제 keep에 저장된 BBox와 많이 겹치는 BBox를 추출해 제거
        # -------------------
        # 하나 줄인 idx까지의 BBox를 out으로 지정한 변수로 생성
        torch.index_select(x1, 0, idx, out=tmp_x1)
        torch.index_select(y1, 0, idx, out=tmp_y1)
        torch.index_select(x2, 0, idx, out=tmp_x2)
        torch.index_select(y2, 0, idx, out=tmp_y2)

        # 모든 BBox에 대해 현재 BBox=index가 i와 겹치는 값까지 설정(clamp)
        tmp_x1 = torch.clamp(tmp_x1, min=x1[i]) #clamp -> 지정된 값이 특정 범위에 있게끔 조정
        tmp_y1 = torch.clamp(tmp_y1, min=y1[i])
        tmp_x2 = torch.clamp(tmp_x2, max=x2[i])
        tmp_y2 = torch.clamp(tmp_y2, max=y2[i])

        # w와 h의 텐서 크기를 index를 1 줄인 크기로 설정
        tmp_w.resize_as_(tmp_x2)
        tmp_h.resize_as_(tmp_y2)

        # clamp된 상태의 BBox의 너비와 높이를 계산
        tmp_w = tmp_x2 - tmp_x1
        tmp_h = tmp_y2 - tmp_y1

        # 너비나 높이가 음수가 된 것은 0으로 설정
        tmp_w = torch.clamp(tmp_w, min=0.0)
        tmp_h = torch.clamp(tmp_h, min=0.0)

        # clamp된 상태의 면적을 계산
        inter = tmp_w * tmp_h

        # IoU = 교집합 부분 / (area(a) + area(b) - 교집합 부분)의 계산
        rem_areas = torch.index_select(area, 0, idx)  # 각 BBox의 원래 면적
        union = (rem_areas - inter) + area[i]  # 두 영역의 합(OR) 면적
        IoU = inter / union

        # IoU가 overlap보다 작은 idx만 남긴다
        idx = idx[IoU.le(overlap)]  # le는 Less than or Equal to 연산 수행
        # IoU가 overlap보다 큰 idx는 처음 선택해 keep에 저장한 idx와 동일한 객체에 대해 BBox를 둘러싼 것이므로 제거

    # while 루프가 종료되면 끝

    return keep, count

```
#### 💬 2.5.3 Detect 클래스 구현
- 입력요소 : 오프셋 정보 `loc`, 신뢰도 `conf`, DBOX정보
** <Forward 흐름> **
1. decode 함수를 이용하여 DBOX정보와 오프셋 정보 loc을 BBOX로 변환
2. conf가 임곗값 이상인 BBOX 추출
3. nm Suprression을 이용하여 중복된 BBOX제거
```
# SSD의 추론 시 conf와 loc의 출력에서 중복을 제거한 BBox를 출력하는 클래스
class Detect(Function):

    def __init__(self, conf_thresh=0.01, top_k=200, nms_thresh=0.45):
        self.softmax = nn.Softmax(dim=-1)  # conf를 소프트맥스 함수로 정규화하기 위해 준비
        self.conf_thresh = conf_thresh  # conf가 conf_thresh=0.01보다 높은 DBox만 처리
        self.top_k = top_k  # nm_supression에서 conf가 높은 top_k개를 계산에 사용, top_k = 200
        self.nms_thresh = nms_thresh  # nm_supression에서 IOU가 nms_thresh=0.45보다 크면 동일 객체에 대한 BBox로 간주

    def forward(self, loc_data, conf_data, dbox_list):
        """
        순전파 계산을 실행.

        Parameters
        ----------
        loc_data:  [batch_num,8732,4]
            오프셋 정보.
        conf_data: [batch_num, 8732,num_classes]
            감지 신뢰도
        dbox_list: [8732,4]
            DBox 정보

        Returns
        -------
        output : torch.Size([batch_num, 21, 200, 5])
            (batch_num, 클래스, conf의 top200, BBox 정보)
        """

        # 각 크기를 얻어옴
        num_batch = loc_data.size(0)  # 미니배치 크기
        num_dbox = loc_data.size(1)  # DBox 개수 = 8732
        num_classes = conf_data.size(2)  # 클래스 개수 = 21

        # conf에 소프트맥스를 적용하여 정규화
        conf_data = self.softmax(conf_data)

        # 출력 형태를 생성. 텐서 크기는 [미니배치 크기, 21, 200, 5]
        #[미니배치 크기, 클래스의 수, 상위 200개중 몇번째인지, 5=신뢰도,xymin max좌표]
        output = torch.zeros(num_batch, num_classes, self.top_k, 5)

        # conf_data를 [batch_num,8732,num_classes]에서 [batch_num, num_classes,8732]로 순서 변경
        conf_preds = conf_data.transpose(2, 1)

        # 미니배치별 루프
        for i in range(num_batch):

            # 1. loc와 DBox에서 수정된 BBox [xmin, ymin, xmax, ymax]를 구함
            decoded_boxes = decode(loc_data[i], dbox_list)

            # conf의 복사본 생성
            conf_scores = conf_preds[i].clone()

            # 이미지 클래스별 루프 (배경 클래스의 인덱스인 0은 계산하지 않고, index=1부터)
            for cl in range(1, num_classes):

                # 2. conf 임계값을 초과한 BBox를 추출
                # conf 임계값을 초과했는지의 마스크를 생성
                # 임계값을 초과한 conf의 인덱스를 c_mask로 가져옴
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                # gt는 Greater than의 약자. gt로 인해 임계값 초과는 1, 이하일 경우 0이 됨
                # conf_scores: torch.Size([21, 8732])
                # c_mask: torch.Size([8732])

                # scores는 torch.Size([임계값 초과 BBox 수])
                scores = conf_scores[cl][c_mask]

                # 임계값을 초과한 conf가 없으면, 즉 scores=[]일 때는 아무것도 하지 않음
                if scores.nelement() == 0:  # nelement로 요소 수 총합 계산
                    continue

                # c_mask를 decoded_boxes에 적용할 수 있도록 크기를 변경
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                # l_mask: torch.Size([8732, 4])

                # l_mask를 decoded_boxes에 적용
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # decoded_boxes[l_mask]로 1차원이 되어버리므로
                # view로 (임계값 초과 BBox 수, 4) 크기로 다시 변형

                # 3. Non-Maximum Suppression을 수행하여 겹치는 BBox 제거
                ids, count = nm_suppression(
                    boxes, scores, self.nms_thresh, self.top_k)
                # ids: conf 내림차순으로 Non-Maximum Suppression을 통과한 인덱스 저장
                # count: Non-Maximum Suppression을 통과한 BBox 수

                # output에 Non-Maximum Suppression을 통과한 결과 저장
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1),
                                                   boxes[ids[:count]]), 1)

        return output  # torch.Size([1, 21, 200, 5])

```

#### 💬 2.5.4 SSD 모델 구현
** <Forward 흐름> **
1. vgg및 extras 모듈을 전달하면서 source1~6 추출
2.source들에 합성곱 층을 각각 한번만 적용하여 오프셋 정보 `loc`와 클래스 신뢰도 `conf` 추출
3. source에서 사용한 DBOX수가 각각 다르므로 텐서모양 변환
4.각각의 변수들을 output에 정리
5.학습 시 output=(loc,conf,dbox_list)
6.추론 시 순전파 함수에 output입력
7.최종 BBOX정보(batch_num,21,200,5) 출력 (학습된 DBOX로 BBOX 정보 생성)
```
# SSD 클래스 생성
class SSD(nn.Module):

    def __init__(self, phase, cfg):
        super(SSD, self).__init__()

        self.phase = phase  # train 또는 inference를 지정
        self.num_classes = cfg["num_classes"]  # 클래스 수 = 21

        # SSD의 네트워크 생성
        self.vgg = make_vgg()
        self.extras = make_extras()
        self.L2Norm = L2Norm()
        self.loc, self.conf = make_loc_conf(
            cfg["num_classes"], cfg["bbox_aspect_num"]
        )

        # DBox 생성
        dbox = DBox(cfg)
        self.dbox_list = dbox.make_dbox_list()

        # 추론 시에는 Detect 클래스를 준비
        if phase == "inference":
            self.detect = Detect()

    def forward(self, x):
        sources = list()  # loc와 conf에 입력되는 source1~6 저장
        loc = list()  # loc의 출력 저장
        conf = list()  # conf의 출력 저장

        # vgg의 conv4_3까지 계산
        for k in range(23):
            x = self.vgg[k](x)

        # conv4_3의 출력을 L2Norm에 입력하여 source1 생성 후 sources에 추가
        source1 = self.L2Norm(x)
        sources.append(source1)

        # vgg를 끝까지 계산하여 source2 생성 후 sources에 추가
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # extras의 conv와 ReLU를 계산
        # source3~6을 sources에 추가
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:  # conv → ReLU → conv → ReLU 후 source에 추가
                sources.append(x)

        # source1~6에 각각 대응하는 합성곱을 1회씩 적용
        # zip으로 다수 리스트의 요소를 동시에 가져옴
        # source1~6이 있으므로 루프가 6회 반복
        for (x, l, c) in zip(sources, self.loc, self.conf):
            # Permute로 요소 순서 변경
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            # l(x)와 c(x)로 합성곱 실행
            # l(x), c(x)의 출력 크기: [batch_num, 4*종횡비의 종류 수, feature map의 높이, feature map의 너비]
            # source에 따라 종횡비 종류 수가 다르므로 순서 변경하여 정렬
            # permute로 순서 변경:
            # [minibatch 수, feature map 수, feature map 수, 4*종횡비 종류 수]
            # (참고)
            # torch.contiguous()는 메모리 상의 요소를 연속적으로 재배치하는 명령
            # 이후 view 함수 사용 시 메모리 연속 배치 필요

        # loc와 conf의 형태 변환
        # loc 크기: torch.Size([batch_num, 34928])
        # conf 크기: torch.Size([batch_num, 183372])
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        # loc와 conf의 형태 재조정
        # loc 크기: torch.Size([batch_num, 8732, 4]) #-1 = 나머지 차원 자동 계산,4=오프셋 정보
        # conf 크기: torch.Size([batch_num, 8732, 21])
        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)

        # 최종 출력
        output = (loc, conf, self.dbox_list)

        if self.phase == "inference":  # 추론 시
            # Detect 클래스의 forward 실행
            # 반환 크기: torch.Size([batch_num, 21, 200, 5])
            return self.detect(output[0], output[1], output[2])

        else:  # 학습 시
            return output
            # 반환값은 (loc, conf, dbox_list)의 튜플

```
### ✏️ 2.6 손실함수 구현
#### 💬 2.6.1 jaccard 계수를 이용한 match 함수의 동작
- 정답 BBOX와 가까운 DBOX를 추출할 때 jaccard 계수 사용
- IOU개념과 동일
**<흐름>**
1. jaccard 계수가 (임계값) 낮은 DBOX를 Negative Dbox로 정의하고 label = 0(배경) 부여
-> 정답 BBox가 없는 Dbox는 물체가 존재하지 않는 배경을 인식하기위해 손실함수의 계산 및 네트워크 학습에 사용

2. jaccard 계수가 (임계값) 높은 Dbox를 Positive Dbox로 정의

3. Positive Dbox를 jaccard 계수가 커지도록 `loc`을 지도 data로 삼고 BBOX가 되게끔 오프셋 학습

4.이 때 DBox의 좌표 정보와 감지되는 클래스를 따로 생각해야한다.
 -> 좌표 오프셋 학습 따로, 객체 분류 따로
 
5. 이 책에서는 모델이 추정한 BBOX와 정답 BBOX간의 jaccard 계수를 처리하지 않고 미리 준비한 DBOX와 정답 BBOX간의 jaccard 계수를 처리한다.

6. 이러한 학습 과정을 `match`를 이용하는데 이 코드 같은 경우에는 저자의 깃헙에 올라와있으며 구현이 아닌 .py 파일에서 import해서 사용한다.
-> 복잡하기 때문

#### 💬 2.6.2 Hard Negative Mining
- 위에서 말했듯이 Negative DBOX는 배경 학습에 사용된다
-> 이 때 학습에 사용되는 N BOX의 수를 줄이는 과정
- 오프셋을 이용한 학습은 Positive Dbox만 사용
-> Positive DBOX는 BBOX가 되기위해 학습해야하기 때문
- 하지만 당연히도 NBOX의 수가 많겠고 이로인해 데이터 불균형(label=0(배경)인 DBOX 수가 많기에)이 발생한다.
- 따라서 N DBOX중 손실 값이 높은 N DBOX들 몇개를 추출해서 그것들을 학습시킨다.

#### 💬 2.6.3 Smooth L1 Loss, Cross Entropy
- Smooth L1 Loss: 지도 데이터와 예측 데이터 간의 차이의 절대값이 1보다 작으면 제곱오차로 처리, 그 외엔 차에서 0.5를 빼고 절댓값
->차이가 큰경우 절대값으로, 차이가 작으면 제곱하여 차이를 극명화

#### 💬 2.6.4 SSD 손실함수 클래스 구현
```
class MultiBoxLoss(nn.Module):
    """SSD의 손실 함수 클래스"""
    
    def __init__(self, jaccard_thresh=0.5, neg_pos=3, device='cpu'):
        super(MultiBoxLoss, self).__init__()
        self.jaccard_thresh = jaccard_thresh  # 0.5 match 함수에서 사용될 jaccard 계수의 임계값
        self.negpos_ratio = neg_pos  # 3:1 Hard Negative Mining에서 음성(배경)과 양성(물체)의 비율
        self.device = device  # CPU 또는 GPU에서 계산

    def forward(self, predictions, targets):
        """
        손실 함수 계산
        
        Parameters
        ----------
        predictions : SSD 네트워크의 학습 시 출력(tuple)
            (loc=torch.Size([num_batch, 8732, 4]), conf=torch.Size([num_batch, 8732, 21]), dbox_list=torch.Size([8732,4])).
        
        targets : [num_batch, num_objs, 5]
            5는 정답 어노테이션 정보 [xmin, ymin, xmax, ymax, label_ind]를 나타냄
        
        Returns
        -------
        loss_l : 텐서
            loc의 손실 값
        loss_c : 텐서
            conf의 손실 값
        """
        
        # SSD 모델의 출력이 튜플이므로 개별적으로 분리
        loc_data, conf_data, dbox_list = predictions
        
        # 요소 수 확인
        num_batch = loc_data.size(0)  # 미니배치 크기
        num_dbox = loc_data.size(1)  # DBox 수 = 8732
        num_classes = conf_data.size(2)  # 클래스 수 = 21
        
        # 손실 계산에 사용할 변수 생성
        # conf_t_label: 각 DBox에 가장 가까운 정답 BBox의 라벨 저장
        # loc_t: 각 DBox에 가장 가까운 정답 BBox의 위치 정보 저장
        conf_t_label = torch.LongTensor(num_batch, num_dbox).to(self.device)
        loc_t = torch.Tensor(num_batch, num_dbox, 4).to(self.device)
        
        # loc_t와 conf_t_label에 DBox와 정답 어노테이션 targets를 match시킨 결과를 덮어씀
        for idx in range(num_batch):  # 미니배치 반복
            # 현재 미니배치의 정답 어노테이션 BBox와 라벨 가져오기
            truths = targets[idx][:, :-1].to(self.device)  # BBox
            labels = targets[idx][:, -1].to(self.device)  # 라벨 [물체1의 라벨, 물체2의 라벨, …]
            
            # 디폴트 박스를 새로운 변수에 준비
            dbox = dbox_list.to(self.device)
            
            # match 함수 실행, loc_t와 conf_t_label 내용 갱신
            # loc_t: 각 DBox에 가장 가까운 정답 BBox의 위치 정보 갱신
            # conf_t_label: 각 DBox에 가장 가까운 BBox의 라벨 갱신
            # 단, 가장 가까운 BBox와의 jaccard overlap이 0.5보다 작으면
            # conf_t_label은 배경 클래스(0)으로 설정
            variance = [0.1, 0.2]  # DBox에서 BBox로 보정 계산 시 사용하는 계수
            match(self.jaccard_thresh, truths, dbox, 
                  variance, labels, loc_t, conf_t_label, idx)
        
        # ---------- 
        # 위치 손실: loss_l 계산
        # Smooth L1 함수로 손실 계산. 단, 물체를 탐지한 DBox의 오프셋만 계산
        # ----------
        pos_mask = conf_t_label > 0  # torch.Size([num_batch, 8732])
        
        # pos_mask를 loc_data 크기로 변형
        pos_idx = pos_mask.unsqueeze(pos_mask.dim()).expand_as(loc_data)
        
        # Positive DBox의 loc_data와 정답 데이터 loc_t 추출
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        
        # 물체를 발견한 Positive DBox의 오프셋 정보 loc_t의 손실 계산
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')
        
        # ----------
        # 클래스 예측 손실: loss_c 계산
        # 교차 엔트로피 손실 함수 사용, 배경 클래스 비율 줄이기 위해 Hard Negative Mining 적용
        # ----------
        batch_conf = conf_data.view(-1, num_classes)
        
        # 클래스 예측 손실 계산 (reduction='none'으로 설정하여 합계를 구하지 않고 차원 유지)
        loss_c = F.cross_entropy(
            batch_conf, conf_t_label.view(-1), reduction='none')
        
        # -----------------
        # Negative DBox 중 Hard Negative Mining으로 추출할 마스크 생성
        # -----------------
        
        # Positive DBox의 손실을 0으로 설정
        num_pos = pos_mask.long().sum(1, keepdim=True)  # 미니배치별 물체 클래스 예측 수
        loss_c = loss_c.view(num_batch, -1)  # torch.Size([num_batch, 8732])
        loss_c[pos_mask] = 0  # 물체를 탐지한 DBox는 손실 0으로 설정
        
        # Hard Negative Mining 적용
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        
        # 배경 DBox 수 num_neg 결정, 물체 탐지 DBox 수의 3배 (self.negpos_ratio배)
        num_neg = torch.clamp(num_pos*self.negpos_ratio, max=num_dbox)
        
        # 손실 값이 큰 순서로 Hard Negative Mining으로 추출
        neg_mask = idx_rank < (num_neg).expand_as(idx_rank)
        
        # -----------------
        # (완료) Hard Negative Mining으로 추출할 Negative DBox의 마스크 생성
        # -----------------
        
        # 마스크 형태를 conf_data에 맞게 조정
        pos_idx_mask = pos_mask.unsqueeze(2).expand_as(conf_data)
        neg_idx_mask = neg_mask.unsqueeze(2).expand_as(conf_data)
        
        # conf_data에서 pos와 neg만 추출하여 conf_hnm 생성
        conf_hnm = conf_data[(pos_idx_mask+neg_idx_mask).gt(0)].view(-1, num_classes)
        
        # 정답 데이터 conf_t_label에서 pos와 neg만 추출하여 conf_t_label_hnm 생성
        conf_t_label_hnm = conf_t_label[(pos_mask+neg_mask).gt(0)]
        
        # confidence 손실 함수 계산 (합계)
        loss_c = F.cross_entropy(conf_hnm, conf_t_label_hnm, reduction='sum')
        
        # 물체를 탐지한 BBox 수 N으로 손실 나눔
        N = num_pos.sum()
        loss_l /= N
        loss_c /= N
        
        return loss_l, loss_c

```
### ✏️ 2.7 학습 및 검증 실시
#### 💬 2.7.1 프로그램 구현
지금까지 작성한 코들을 이용하여 학습 프로그램을 구현해보자
**<흐름>**
1. 데이터 로더 만들기
2. 네트워크 모델 만들기
3. 손실함수 정의
4. 최적화 기법 설정
5. 학습 및 검증 실시
#### 💬 2.7.2 데이터 로더 만들기
```
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("사용 장치:", device)

from utils.ssd_model import make_datapath_list, VOCDataset, DataTransform, Anno_xml2list, od_collate_fn

# 파일 경로 리스트 가져오기
rootpath='/content/drive/MyDrive/Colab Notebooks/pytorch_advanced/2_objectdetection/data/VOCdevkit/VOC2012/'
train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath)

# 데이터셋 생성
voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']
color_mean = (104, 117, 123)  # (BGR) 색상 평균값
input_size = 300  # 입력 이미지 크기를 300×300으로 설정

train_dataset = VOCDataset(train_img_list, train_anno_list, phase="train",
                           transform=DataTransform(input_size, color_mean),
                           transform_anno=Anno_xml2list(voc_classes))

val_dataset = VOCDataset(val_img_list, val_anno_list, phase="val",
                         transform=DataTransform(input_size, color_mean),
                         transform_anno=Anno_xml2list(voc_classes))

# DataLoader 생성
batch_size = 32

train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=od_collate_fn)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=od_collate_fn)

# 딕셔너리 객체로 묶기
dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

```

#### 💬 2.7.3 네트워크 모델 만들기
- 초깃값 He 추가(활성화 함수 Relu일때 사용)
- 학습된 모듈 사용
```
from utils.ssd_model import SSD

# SSD300 설정
ssd_cfg = {
    'num_classes': 21,  # 배경 클래스를 포함한 전체 클래스 수
    'input_size': 300,  # 입력 이미지 크기
    'bbox_aspect_num': [4, 6, 6, 6, 4, 4],  # 출력할 DBox의 종횡비 종류
    'feature_maps': [38, 19, 10, 5, 3, 1],  # 각 source의 이미지 크기
    'steps': [8, 16, 32, 64, 100, 300],  # DBox 크기 결정
    'min_sizes': [30, 60, 111, 162, 213, 264],  # DBox 최소 크기
    'max_sizes': [60, 111, 162, 213, 264, 315],  # DBox 최대 크기
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],  # DBox의 종횡비
}

# SSD 네트워크 모델
net = SSD(phase="train", cfg=ssd_cfg)

# SSD 초기 가중치 설정
# SSD의 VGG 부분에 가중치 로드
vgg_weights = torch.load('./weights/vgg16_reducedfc.pth')
net.vgg.load_state_dict(vgg_weights)

# SSD의 나머지 네트워크 가중치는 He 초기값으로 초기화
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:  # 바이어스 항이 있는 경우
            nn.init.constant_(m.bias, 0.0)

# He 초기값 적용
net.extras.apply(weights_init)
net.loc.apply(weights_init)
net.conf.apply(weights_init)

# GPU 사용 여부 확인
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("사용 중인 디바이스:", device)

print('네트워크 설정 완료: 학습된 가중치를 로드했습니다.')

```
#### 💬 2.7.4 손실함수 및 최적화 기법 설정
```
from utils.ssd_model import MultiBoxLoss
import torch.optim as optim

# 손실함수 설정
criterion = MultiBoxLoss(jaccard_thresh=0.5, neg_pos=3, device=device)

# 최적화 설정
optimizer = optim.SGD(net.parameters(), lr=1e-3,
                      momentum=0.9, weight_decay=5e-4)

```
#### 💬 2.7.5 학습 및 검증 실시
```
import time
# 모델 학습 함수 정의
def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
    # GPU 사용 여부 확인
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("사용 중인 디바이스:", device)

    # 네트워크를 GPU로 이동
    net.to(device)

    # 네트워크가 고정되어 있다면 속도 최적화
    torch.backends.cudnn.benchmark = True

    # 반복 카운터 설정
    iteration = 1
    epoch_train_loss = 0.0  # 에폭의 훈련 손실 합계
    epoch_val_loss = 0.0  # 에폭의 검증 손실 합계
    logs = []

    # 에폭 루프
    for epoch in range(num_epochs + 1):
        # 시작 시간 저장
        t_epoch_start = time.time()
        t_iter_start = time.time()

        print('-------------')
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-------------')

        # 에폭별 훈련 및 검증 루프
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # 모델을 훈련 모드로 전환
                print('(train)')
            else:
                if (epoch + 1) % 10 == 0:
                    net.eval()  # 모델을 검증 모드로 전환
                    print('-------------')
                    print('(val)')
                else:
                    # 검증은 10번의 에폭마다 한 번 실행
                    continue

            # DataLoader에서 미니배치를 반복적으로 가져오기
            for images, targets in dataloaders_dict[phase]:
                # 데이터를 GPU로 이동
                images = images.to(device)
                targets = [ann.to(device) for ann in targets]  # 리스트 요소를 GPU로 이동

                # optimizer 초기화
                optimizer.zero_grad()

                # 순전파(forward) 계산
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(images)  # 순전파 계산
                    loss_l, loss_c = criterion(outputs, targets)  # 손실 계산
                    loss = loss_l + loss_c

                    # 훈련 시 역전파 실행
                    if phase == 'train':
                        loss.backward()  # 기울기 계산

                        # 기울기 클리핑: 최대값을 2.0으로 제한하여 안정성 확보
                        nn.utils.clip_grad_value_(net.parameters(), clip_value=2.0)

                        optimizer.step()  # 파라미터 업데이트

                        if iteration % 10 == 0:  # 10번 반복마다 손실 출력
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            print(f'반복 {iteration} || 손실: {loss.item():.4f} || 10반복 소요 시간: {duration:.4f} 초')
                            t_iter_start = time.time()

                        epoch_train_loss += loss.item()
                        iteration += 1
                    else:  # 검증 시
                        epoch_val_loss += loss.item()

        # 에폭의 phase별 손실 출력
        t_epoch_finish = time.time()
        print('-------------')
        print(f'에폭 {epoch + 1} || 훈련 손실: {epoch_train_loss:.4f} || 검증 손실: {epoch_val_loss:.4f}')
        print(f'에폭 소요 시간: {t_epoch_finish - t_epoch_start:.4f} 초')

        # 로그 저장
        log_epoch = {'epoch': epoch + 1, 'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss}
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv("log_output.csv")

        epoch_train_loss = 0.0  # 에폭 훈련 손실 초기화
        epoch_val_loss = 0.0  # 에폭 검증 손실 초기화

        # 네트워크 저장
        if (epoch + 1) % 10 == 0:
            torch.save(net.state_dict(), f'weights/ssd300_{epoch + 1}.pth')

```
- 책에선 50회, 원 논문에선 5만회를 학습 시킨다. 50회만 돌려도 6시간이 걸린다고 한다.
본인은 1epoch만 학습 시켰고 결과는 아래와 같다.
```
사용 중인 디바이스: cuda:0
-------------
Epoch 1/1
-------------
(train)
반복 10 || 손실: 16.5876 || 10반복 소요 시간: 241.5497 초
반복 20 || 손실: 12.9065 || 10반복 소요 시간: 200.3242 초
반복 30 || 손실: 10.5291 || 10반복 소요 시간: 196.8386 초
반복 40 || 손실: 9.1139 || 10반복 소요 시간: 198.0570 초
반복 50 || 손실: 8.0722 || 10반복 소요 시간: 199.7148 초
반복 60 || 손실: 8.0760 || 10반복 소요 시간: 199.3600 초
반복 70 || 손실: 8.2441 || 10반복 소요 시간: 201.6204 초
반복 80 || 손실: 7.5729 || 10반복 소요 시간: 195.7151 초
반복 90 || 손실: 7.9020 || 10반복 소요 시간: 197.2062 초
반복 100 || 손실: 7.5496 || 10반복 소요 시간: 198.7219 초
반복 110 || 손실: 7.2365 || 10반복 소요 시간: 201.1805 초
반복 120 || 손실: 7.0479 || 10반복 소요 시간: 195.2745 초
반복 130 || 손실: 7.5754 || 10반복 소요 시간: 199.3897 초
반복 140 || 손실: 7.5078 || 10반복 소요 시간: 196.9685 초
반복 150 || 손실: 7.0848 || 10반복 소요 시간: 196.5162 초
반복 160 || 손실: 7.4417 || 10반복 소요 시간: 200.6713 초
반복 170 || 손실: 7.1285 || 10반복 소요 시간: 205.0768 초
-------------
에폭 1 || 훈련 손실: 1613.8578 || 검증 손실: 0.0000
에폭 소요 시간: 3632.3862 초
-------------
Epoch 2/1
-------------
(train)
반복 180 || 손실: 7.3258 || 10반복 소요 시간: 1.1452 초
반복 190 || 손실: 7.2704 || 10반복 소요 시간: 16.3383 초
반복 200 || 손실: 6.9716 || 10반복 소요 시간: 17.1263 초
반복 210 || 손실: 6.5950 || 10반복 소요 시간: 17.0158 초
반복 220 || 손실: 6.7655 || 10반복 소요 시간: 16.6718 초
반복 230 || 손실: 6.9900 || 10반복 소요 시간: 17.3153 초
반복 240 || 손실: 6.8913 || 10반복 소요 시간: 16.6899 초
반복 250 || 손실: 6.9345 || 10반복 소요 시간: 16.6941 초
반복 260 || 손실: 6.8544 || 10반복 소요 시간: 17.2184 초
반복 270 || 손실: 7.1015 || 10반복 소요 시간: 17.3236 초
반복 280 || 손실: 6.5622 || 10반복 소요 시간: 16.7740 초
반복 290 || 손실: 6.6835 || 10반복 소요 시간: 17.1392 초
반복 300 || 손실: 6.2716 || 10반복 소요 시간: 17.9147 초
반복 310 || 손실: 6.8892 || 10반복 소요 시간: 16.7970 초
반복 320 || 손실: 6.6352 || 10반복 소요 시간: 16.9965 초
반복 330 || 손실: 7.0024 || 10반복 소요 시간: 17.4508 초
반복 340 || 손실: 6.3415 || 10반복 소요 시간: 16.8674 초
반복 350 || 손실: 6.8186 || 10반복 소요 시간: 17.4949 초
-------------
에폭 2 || 훈련 손실: 1243.7942 || 검증 손실: 0.0000
에폭 소요 시간: 315.3779 초
```
#### 💬 2.8.1 추론
학습된 모델을 사용한다
```
from utils.ssd_model import SSD

# VOC 클래스 목록
voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

# SSD300 설정
ssd_cfg = {
    'num_classes': 21,  # 배경 클래스를 포함한 총 클래스 수
    'input_size': 300,  # 이미지 입력 크기
    'bbox_aspect_num': [4, 6, 6, 6, 4, 4],  # 출력할 DBox의 종횡비 종류
    'feature_maps': [38, 19, 10, 5, 3, 1],  # 각 source의 이미지 크기
    'steps': [8, 16, 32, 64, 100, 300],  # DBox의 크기를 결정
    'min_sizes': [30, 60, 111, 162, 213, 264],  # DBox의 최소 크기
    'max_sizes': [60, 111, 162, 213, 264, 315],  # DBox의 최대 크기
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
}

# SSD 네트워크 모델 생성 (추론 모드)
net = SSD(phase="inference", cfg=ssd_cfg)

# SSD의 학습된 가중치 설정
# net_weights = torch.load('./weights/ssd300_50.pth',
#                          map_location={'cuda:0': 'cpu'})
net_weights = torch.load('./weights/ssd300_mAP_77.43_v2.pth',
                         map_location={'cuda:0': 'cpu'})

net.load_state_dict(net_weights)

print('네트워크 설정 완료: 학습된 가중치를 로드했습니다')

```
```
from utils.ssd_model import DataTransform

# 1. 이미지 읽기
image_file_path = "./data/cowboy-757575_640.jpg"
img = cv2.imread(image_file_path)  # [높이][너비][색상(BGR)]
height, width, channels = img.shape  # 이미지 크기 가져오기

# 2. 원본 이미지 표시
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# 3. 전처리 클래스 생성
color_mean = (104, 117, 123)  # (BGR) 평균 색상 값
input_size = 300  # 이미지 입력 크기를 300×300으로 설정
transform = DataTransform(input_size, color_mean)

# 4. 전처리 수행
phase = "val"
img_transformed, boxes, labels = transform(img, phase, "", "")  # 어노테이션은 없으므로 ""로 처리
img = torch.from_numpy(img_transformed[:, :, (2, 1, 0)]).permute(2, 0, 1)

# 5. SSD로 예측
net.eval()  # 네트워크를 추론 모드로 설정
x = img.unsqueeze(0)  # 미니배치화: torch.Size([1, 3, 300, 300])
detections = net(x)

print(detections.shape)
print(detections)
# 출력: torch.Size([batch_num, 21, 200, 5])
# = (batch_num, 클래스 수, conf 상위 200개, 정규화된 BBox 정보)
# 정규화된 BBox 정보 (확신도, xmin, ymin, xmax, ymax)

```

BBOX 그리기
```
# 이미지에 대한 예측
from utils.ssd_predict_show import SSDPredictShow

# 파일 경로
image_file_path = "./data/cowboy-757575_640.jpg"

# 예측과 예측 결과를 이미지에 그리기
ssd = SSDPredictShow(eval_categories=voc_classes, net=net)
ssd.show(image_file_path, data_confidence_level=0.6)

```

## 🔎 2장을 마치며
2장부터는 저자가 제공한 .py파일에서 import하여 black box로 사용하는 코드들이 많았고 저자 또한 코드보다 이런 SSD 학습의 flow를 보라고 해서 1장에 비해 흐름을 중심으로 봤다.
