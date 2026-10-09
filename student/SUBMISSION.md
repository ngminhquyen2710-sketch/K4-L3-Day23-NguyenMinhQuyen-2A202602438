# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Nguyễn Minh Quyền
- MSSV: 2A202602438
- Email: ngminhquyen2710@gmail.com
- Link repo (fork): https://github.com/ngminhquyen2710-sketch/K4-L3-Day23-NguyenMinhQuyen-2A202602438.git
- Commit hash nộp (`git rev-parse HEAD`): 072b9e49e777918c218c7cca0a36a64d6d64d8f0

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`: compare, [0, 198], training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord, 0
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`: precision=0.9701 (97.01%), recall=0.7004 (70.04%), tp=519, fp=16, fn=222
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: rmse=0.1503 m, matches=502, sum_sq_err=11.3369 m², ghost_track_frames=0, missed_gt_frames=239, mean_confirmed_tracks=2.5226
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: rmse=0.1358 m, matches=502, sum_sq_err=9.2642 m², ghost_track_frames=0, missed_gt_frames=239, mean_confirmed_tracks=2.5226
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss: Cả hai chế độ đều có số cặp ghép bằng nhau (matches = 502), hoàn toàn không có track ma (ghost_track_frames = 0) và số lượng nhãn GT bị bỏ sót là như nhau (missed_gt_frames = 239, trung bình 2.52 confirmed tracks/frame). Điều này đúng với thiết kế track-then-fuse vì Camera không tham gia tạo hay xóa track, số lượng và vòng đời track hoàn toàn do LiDAR định đoạt. Về độ chính xác vị trí, chế độ Fused đạt RMSE = 0.1358 m, thấp hơn chế độ LiDAR-only (0.1503 m), giảm tổng bình phương sai số từ 11.34 m² xuống 9.26 m² (giảm ~18.3%). Điều này chứng minh đo lường bổ sung từ Camera kết hợp Jacobian mô hình pinhole trong EKF đã giúp tinh chỉnh trạng thái tốt hơn mà không gây suy giảm chất lượng theo dõi (rmse_fused - rmse_lidar = -0.0145 m <= 0.05 m).

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?
- Vector đo z: LiDAR đo trực tiếp vị trí tâm xe 3D trong không gian thực z = [x, y, z]^T (đơn vị: mét) với hàm đo tuyến tính h(x) = Hx và H là ma trận cố định 3x6. Camera đo tọa độ điểm ảnh 2D z = [u, v]^T (đơn vị: pixel) thông qua phép chiếu pinhole phi tuyến h(x) phụ thuộc vào tọa độ xe trong hệ cảm biến (u = c_i - f_i*y_s/x_s, v = c_j - f_j*z_s/x_s), do đó ma trận H là Jacobian 2x6 biến thiên theo trạng thái.
- Ma trận hiệp phương sai nhiễu đo R: LiDAR có R là ma trận đường chéo 3x3 với sigma_x = sigma_y = sigma_z = 0.1 m (đơn vị m^2). Camera có R là ma trận đường chéo 2x2 với sigma_u = sigma_v = 5.0 px (đơn vị pixel^2).

2. Vì sao cần gating Mahalanobis trước khi gán?
Khoảng cách Mahalanobis bình phương d^2 = gamma^T * S^(-1) * gamma chuẩn hóa độ lệch giữa dự báo và đo lường theo ma trận hiệp phương sai bất định S = H P H^T + R. Cổng chi-square (chi2_gate) loại bỏ các quan sát ngoại lai (outliers) hoặc vật thể khác nằm ngoài vùng phân bố xác suất tin cậy của track trước khi ghép cặp. Việc này ngăn chặn gán nhầm đối tượng (cross-track association) và giảm không gian tìm kiếm của thuật toán greedy.

3. Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`.
Pipeline là **track-then-fuse** (ở cấp độ đối tượng / track-level). Hệ thống chỉ duy trì một danh sách track duy nhất: mỗi frame EKF predict 1 lần, sau đó gán và update EKF bằng LiDAR (AssocL), rồi tiếp tục gán và update EKF bằng Camera (AssocC) nếu xe nằm trong tầm nhìn camera. Trên log `grade_run.log` và code `run_lab.py`, thứ tự mỗi frame luôn là: `KF.predict(track)` -> `associate_and_update(..., lidar)` -> `associate_and_update(..., camera)`, trong đó camera chỉ refine trạng thái của các track đã có chứ không tạo track mới.

4. Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?
Nếu camera lệch calibration (sai lệch extrinsic hoặc intrinsic), hàm chiếu h(x) sẽ tính ra tọa độ pixel dự báo lệch có hệ thống so với điểm đo thực tế. Triệu chứng là vector innovation gamma = z - h(x) sẽ xuất hiện độ lệch kỳ vọng khác 0 (systematic bias). Khoảng cách Mahalanobis tăng cao khiến phép đo bị cổng chi-square từ chối liên tục (miss association). Nếu vẫn bị gán, EKF update sẽ kéo sai trạng thái ước lượng, làm tăng vọt sai số vị trí RMSE hoặc làm bộ lọc phân kỳ.

5. Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng?
   Giải thích vì sao lidar quyết định score/init/delete còn camera chỉ EKF update.
- Cần sensor tường minh ở frame rỗng vì cuối hàm luôn phải gọi `manager.manage_tracks(...)`. Nếu là lượt LiDAR, mọi track trong FOV không có đo sẽ bị tính là miss và bị trừ điểm existence score (-1/window) cũng như kiểm tra xóa track. Nếu là lượt Camera, frame rỗng sẽ bỏ qua mà không làm ảnh hưởng đến điểm số hay số lượng track.
- LiDAR quyết định lifecycle vì LiDAR cung cấp tọa độ 3D đầy đủ trong không gian thực và bao quát góc nhìn rộng quanh xe, đủ tin cậy để khởi tạo và quyết định sự tồn tại của vật thể 3D. Camera chỉ là cảm biến 2D không đo trực tiếp độ sâu và có góc nhìn hẹp (chỉ FRONT), nếu cho phép Camera can thiệp lifecycle sẽ gây mơ hồ độ sâu hoặc xóa oan các track khi xe đi ra ngoài khung hình camera phía trước.

6. Nêu điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xóa track.
- Xác nhận track: Track đạt điều kiện `score > confirmed_threshold` (0.8), tương đương với 5 hit liên tiếp trong cửa sổ window = 6.
- Giữ confirmed sau miss: Khi track đã ở trạng thái confirmed, nếu bị miss trong FOV LiDAR thì điểm chỉ bị trừ 1/window (ví dụ còn 5/6 = 0.833) nhưng trạng thái vẫn được giữ nguyên là `confirmed` (không bị giáng cấp về tentative).
- Xóa track (`should_delete_track`): Xóa khi phương sai vị trí P[0,0] > max_P (9.0) hoặc P[1,1] > max_P; HOẶC track confirmed có `score < delete_threshold` (0.6); HOẶC track chưa confirmed có `score <= 0`.

## Bonus (không bắt buộc)

Liệt kê phần bonus đã làm, file bằng chứng trong `student/bonus/` và kết quả chính
(xem [RUBRIC.md](../RUBRIC.md) mục 2). Không làm thì ghi "Không".

- Không

## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Antigravity AI Assistant (Gemini 3.8 Flash)
- Dùng cho phần nào (hàm, câu hỏi, debug): Triển khai mã nguồn Part E (EKF), Part G (mô hình camera pinhole), Part F (association & gating), Part H (track lifecycle); hỗ trợ giải thích các câu hỏi lý thuyết và tự động hóa kiểm thử.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): Kiểm tra toàn bộ 128/128 test trong suite pytest student/tests đạt pass 100%, chạy thử nghiệm fusion-run-lab trên toàn bộ 198 frame của segment Waymo đạt RMSE <= 0.15 m, đối chiếu chặt chẽ với tài liệu hướng dẫn kỹ thuật docs/HUONG_DAN_KY_THUAT.md.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: không bắt buộc sửa (hoặc ghi chú nếu bạn đã sửa)
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [x] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [x] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
