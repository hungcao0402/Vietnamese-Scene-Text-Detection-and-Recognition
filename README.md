
# BKAI-NAVER 2022 - Track 3: OCR - Team: UIT.TurtleDog

Lưu ý: 

- Trong quá trình chạy, hầu hết chúng tôi đều sử dụng bash scripts để tiện lợi trong quá trình chạy. Chúng tôi có comment mục đích các lệnh mà chúng tôi chạy trong các file bash scripts này để hiểu rõ hơn.

- Code này là version1 chỉ sử dụng 1 detector là ABCNetV2 ở vòng public test, version2 có thêm Yolov5 sẽ được update sau.

## 1. Setup môi trường:

Để thuận tiện và tránh xung đột trong việc cài đặt, chúng tôi sử dụng Docker chạy Linux dùng làm công cụ để cài đặt.

Nhằm cài đặt Docker thích hợp cho môi trường máy host, mọi người có thể theo các hướng dẫn tại trang chủ của Docker.

Sau khi unzip toàn bộ source code của chúng tôi, hãy tiến hành build Docker image.

```
cd docker
docker build -t <tên image> .
```

Khi build xong, hãy vào file `run_docker.sh` để chỉnh sửa tên image, đường dẫn nhằm mount folder code của chúng tôi với thư mục trong container và số GPUs muốn sử dụng (xem dòng lệnh chúng tôi đã để sẵn).

Sau đó, tiến hành start container:
```
bash run_docker.sh
```

Sau khi đã vào được bên trong container, tiến hành cài đặt các thư viện cần thiết.

```
pip3 install -r requirements.txt
```

## 2. Download resources cần thiết.

Trong quá trình tạo dữ liệu cho training, chúng tôi có các resources cần sử dụng:

- `bg_img.zip`: Bộ 8,000 background images từ repo [SynthText](https://github.com/ankush-me/SynthText)
- `resources.zip`: Gồm 2,000 background images chúng tôi lọc ra từ bộ SynthText trên; các fonts cần sử dụng trong quá trình sinh data; tập corpus từ điển Tiếng Việt được chúng tôi lấy từ repo [viwik18](https://github.com/NTT123/viwik18); Đồng thời, khâu sinh tập colormap của background images bởi SynthTiger (phương pháp chúng tôi sử dụng để sinh data) khá tốn thời gian nên chúng tôi cung cấp trước tập colormap này.

Tất cả các files trên được chúng tôi để ở trên thư mục Drive [này](https://drive.google.com/drive/folders/1Dt4jd79_WODHtD7mkWF0TnHNs-0BXKVs?usp=sharing).

Chúng tôi cũng cung cấp code để download và unzip các data trên và data từ ban tổ chức vào đúng vị trí. Mọi người có thể chạy:

```
bash scripts/download.sh
```

## 3. Chuẩn bị dữ liệu

Trước tiên, tiến hành xử lý dữ liệu từ BTC.
```
bash scripts/prepare_data.sh
```

Sau đó, chạy các file scripts sau để tiến hành sinh dữ liệu

```
cd prepare_data
bash prepare_det_data.sh
bash prepare_rec_data.sh
```

## 3. Train

### a. Detection - ABCnetV2

Chạy câu lệnh sau:

```
cd ABCnetV2
python3 setup.py build develop
```

Trước khi train, xem file `train.sh` để điều chỉnh các tham số về cấu hình train như `num-gpus`, `num-machine`,... cho phù hợp với cấu hình.

Một vài lưu ý trước khi train:

- Số lượng ảnh ở mỗi batch (tức batch size) cần phải chia hết cho số lượng GPUs dùng để train. Để thiết lập thông số này, vào file `ABCnetV2/configs/BAText/VinText/v2_attn_R_50.yaml` và điều chỉnh biến `IMS_PER_BATCH`.

- Với kết quả hiện tại tốt nhất của team đạt được, chúng tôi train với config như sau:

    + Tổng thời gian train: 15.57hrs

    + GPUs: 3 x GPUs (NVIDIA GeForce RTX 2080 Ti 12GB)

    + Số iteration: 80,000

Output sau khi train (file log, weights,...) sẽ được đặt ở thư mục `ABCnetV2/output/ABCnetV2`

### b. Recognition - Iternet
```
cd iternet
bash train.sh
```
Output sau khi train (file log, weights,...) sẽ được đặt ở thư mục 'iternet/workdir'

## 4. Chạy đánh giá

Trước khi tiến hành đánh giá, vui lòng để toàn bộ hình ảnh cần được đánh giá vào thư mục `data/public_test_images` (chúng tôi đã có down sẵn tập test A ở file `download.sh`).

Sau đó tiến hành chạy đánh giá model detection:

```
cd ABCnetV2
bash detect.sh
```

Tiếp theo, chạy đánh giá model recognition

```
cd iternet
bash predict.sh
```

Kết quả cuối cùng sẽ là file `iternet/sub.zip`.






