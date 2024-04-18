from PyQt5.QtWidgets import *
# from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtGui import QPainter, QPen, QColor, QImage, QPixmap
from PyQt5.QtCore import Qt, QPointF
import numpy as np
import sys
import time
import cv2
import os
from UI import Ui_MainWindow
import matplotlib.pyplot as plt
import pandas as pd
from argparse import ArgumentParser
from argparse import ArgumentParser
import cv2
import numpy as np
from lib.cv_thread import VideoToImagesThread
from lib.util import DataType
from lib.timer import Timer
from lib.vis_pose import draw_points_and_skeleton, joints_dict
from Widget.store import Store_Widget
from topdown_demo_with_mmdet import process_one_image
from mmengine.logging import print_log
import sys
sys.path.append("c:\\users\\chenbo\\desktop\\skeleton2d\\src\\tracker")
from pathlib import Path
from tracker.mc_bot_sort import BoTSORT
from tracker.tracking_utils.timer import Timer
from mmpose.apis import init_model as init_pose_estimator
from mmpose.utils import adapt_mmdet_pipeline
from lib.one_euro_filter import OneEuroFilter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


class Pose_2d_Tab_Control(QMainWindow):
    def __init__(self):
        super(Pose_2d_Tab_Control, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.init_var()
        self.bind_ui()
        self.add_parser()
        self.set_tracker_parser()
        self.init_model()

    def bind_ui(self):
        self.clear_table_view()
        self.ui.load_video_btn.clicked.connect(
            lambda: self.load_video(self.ui.video_label, self.db_path + "/videos/"))
        # self.ui.store_video_btn.clicked.connect(self.store_video)
        self.ui.play_btn.clicked.connect(self.play_btn_clicked)
        self.ui.back_key_btn.clicked.connect(
            lambda: self.ui.frame_slider.setValue(self.ui.frame_slider.value() - 1)
        )
        self.ui.forward_key_btn.clicked.connect(
            lambda: self.ui.frame_slider.setValue(self.ui.frame_slider.value() + 1)
        )
        self.ui.frame_slider.valueChanged.connect(self.analyze_frame)
        self.ui.ID_selector.currentTextChanged.connect(lambda: self.import_data_to_table(self.ui.frame_slider.value()))
        self.ui.correct_btn.clicked.connect(self.update_person_df)
        self.ui.show_skeleton_checkBox.setChecked(True)
        self.ui.ID_Locker.clicked.connect(self.ID_locker)
        self.ui.Keypoint_Table.cellActivated.connect(self.on_cell_clicked)
        self.ui.Frame_View.mousePressEvent = self.mousePressEvent
        self.ui.store_data_btn.clicked.connect(self.show_store_window)
        self.ui.li_btn.clicked.connect(self.linear_interpolation)
        self.ui.load_data_btn.clicked.connect(self.load_json)
        self.ui.smooth_data_btn.clicked.connect(self.smooth_data)
        self.ui.start_code_btn.clicked.connect(self.start_analyze_frame)

    def init_model(self):
        self.detector = init_detector(
        self.args.det_config, self.args.det_checkpoint)
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)
        self.pose_estimator = init_pose_estimator(
        self.args.pose_config,
        self.args.pose_checkpoint,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=self.args.draw_heatmap))))
        self.tracker = BoTSORT(self.tracker_args, frame_rate=30.0)
        self.smooth_tool = OneEuroFilter()
        self.timer = Timer()

    def init_var(self):
        self.db_path = f"../../Db"
        self.is_paly = False
        self.is_play=False
        self.processed_images=-1
        self.fps = 30
        self.video_images=[]
        self.video_path = ""
        self.is_threading=False
        self.video_scene = QGraphicsScene()
        self.video_scene.clear()
        self.correct_kpt_idx = 0
        self.video_name = ""
        
        self.processed_frames = set()
        self.person_df = pd.DataFrame()
        self.person_data = []
        self.label_kpt = False
        self.ID_lock = False
        self.select_id = 0
        self.select_kpt_index = 0

        self.kpts_dict = joints_dict()['coco']['keypoints']
        try:
            self.colors = np.round(
                np.array(plt.get_cmap('gist_rainbow').colors) * 255
            ).astype(np.uint8)[:, ::-1].tolist()
        except AttributeError:  # if palette has not pre-defined colors
            self.colors = np.round(
                np.array(plt.get_cmap('gist_rainbow')(np.linspace(0, 1, 10))) * 255
            ).astype(np.uint8)[:, -2::-1].tolist()
            
    def add_parser(self):
        self.parser = ArgumentParser()
        self.parser.add_argument('--det-config', default='../mmpose_main/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py', help='Config file for detection')
        self.parser.add_argument('--det-checkpoint', default='../../Db/pretrain/vit_pose_pth/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth', help='Checkpoint file for detection')
        self.parser.add_argument('--pose-config', default='../mmpose_main/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192.py', help='Config file for pose')
        self.parser.add_argument('--pose-checkpoint', default='../../Db/pretrain/vit_pose_pth/210/work_dirs/td-hm_ViTPose-base_8xb64-210e_coco-256x192/epoch_210.pth', help='Checkpoint file for pose')
        self.parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
        self.parser.add_argument(
            '--bbox-thr',
            type=float,
            default=0.3,
            help='Bounding box score threshold')
        self.parser.add_argument(
            '--nms-thr',
            type=float,
            default=0.3,
            help='IoU threshold for bounding box NMS')
        self.parser.add_argument(
            '--kpt-thr',
            type=float,
            default=0.3,
            help='Visualizing keypoint thresholds')
        self.parser.add_argument(
            '--draw-heatmap',
            action='store_true',
            default=False,
            help='Draw heatmap predicted by the model')
        self.parser.add_argument(
            '--show-kpt-idx',
            action='store_true',
            default=False,
            help='Whether to show the index of keypoints')
        self.parser.add_argument(
            '--skeleton-style',
            default='mmpose',
            type=str,
            choices=['mmpose', 'openpose'],
            help='Skeleton style selection')
        self.parser.add_argument(
            '--radius',
            type=int,
            default=3,
            help='Keypoint radius for visualization')
        self.parser.add_argument(
            '--thickness',
            type=int,
            default=1,
            help='Link thickness for visualization')
        self.parser.add_argument(
            '--show-interval', type=int, default=0, help='Sleep seconds per frame')
        self.parser.add_argument(
            '--alpha', type=float, default=0.8, help='The transparency of bboxes')
        self.parser.add_argument(
            '--draw-bbox', action='store_true', help='Draw bboxes of instances')
        self.args = self.parser.parse_args()

    def set_tracker_parser(self):
        parser = ArgumentParser()
        # tracking args
        parser.add_argument("--track_high_thresh", type=float, default=0.3, help="tracking confidence threshold")
        parser.add_argument("--track_low_thresh", default=0.05, type=float, help="lowest detection threshold")
        parser.add_argument("--new_track_thresh", default=0.4, type=float, help="new track thresh")
        parser.add_argument("--track_buffer", type=int, default=360, help="the frames for keep lost tracks")
        parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
        parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                            help="threshold for filtering out boxes of which aspect ratio are above the given value.")
        parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
        parser.add_argument("--fuse-score", dest="mot20", default=True, action='store_true',
                            help="fuse score and iou for association")

        # CMC
        parser.add_argument("--cmc-method", default="sparseOptFlow", type=str, help="cmc method: sparseOptFlow | files (Vidstab GMC) | orb | ecc")

        # ReID
        parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="with ReID module.")
        parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml",
                            type=str, help="reid config file path")
        parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth",
                            type=str, help="reid config file path")
        parser.add_argument('--proximity_thresh', type=float, default=0.5,
                            help='threshold for rejecting low overlap reid matches')
        parser.add_argument('--appearance_thresh', type=float, default=0.25,
                            help='threshold for rejecting low appearance similarity reid matches')

        self.tracker_args = parser.parse_args()

        self.tracker_args.jde = False
        self.tracker_args.ablation = False

    def load_video(self, label_item, path):
        if self.is_play:
            QMessageBox.warning(self, "讀取影片失敗", "請先停止播放影片!")
            return
        self.init_var()
        self.video_path = self.load_data(
                label_item, path, None, DataType.VIDEO)       
        # no video found
        if self.video_path == "":
            return
        label_item.setText("讀取影片中...")
        #run image thread
        self.v_t=VideoToImagesThread(self.video_path)
        self.v_t.emit_signal.connect(self.video_to_frame)
        self.v_t.start()

    def load_data(self, label_item, dir_path="", value_filter=None, mode=DataType.DEFAULT):
        data_path = None
        if mode == DataType.FOLDER:
            data_path = QFileDialog.getExistingDirectory(
                self, mode.value['tips'], dir_path)
        else:
            name_filter = mode.value['filter'] if value_filter == None else value_filter
            data_path, _ = QFileDialog.getOpenFileName(
                None, mode.value['tips'], dir_path, name_filter)
        if label_item == None:
            return data_path
        # check exist
        if data_path:
            label_item.setText(os.path.basename(data_path))
            label_item.setToolTip(data_path)
        else:
            label_item.setText(mode.value['tips'])
            label_item.setToolTip("")
        return data_path  

    def show_image(self, image: np.ndarray, scene: QGraphicsScene, GraphicsView: QGraphicsView): 
        scene.clear()
        image = image.copy()
        w, h = image.shape[1], image.shape[0]
        bytesPerline = 3 * w
        qImg = QImage(image, w, h, bytesPerline, QImage.Format_RGB888).rgbSwapped()   
        pixmap = QPixmap.fromImage(qImg)
        scene.addPixmap(pixmap)
        GraphicsView.setScene(scene)
        GraphicsView.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

    def video_to_frame(self, video_images, fps, count):
        self.total_images = count
        self.ui.frame_slider.setMinimum(0)
        self.ui.frame_slider.setMaximum(count - 1)
        self.ui.frame_slider.setValue(0)
        self.ui.frame_num_label.setText(f'0/{count-1}')
        # show the first image of the video
        self.video_images=video_images
        self.show_image(self.video_images[0], self.video_scene, self.ui.Frame_View)
        self.video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        self.ui.video_label.setText(self.video_name)
        height, width, _ = self.video_images[0].shape
        self.close_thread(self.v_t)
        self.fps = fps

    def close_thread(self, thread):
        thread.stop()
        thread = None
        self.is_threading=False
   
    def play_frame(self, start_num=0):
        for i in range(start_num, self.total_images):
            if not self.is_play:
                break
            if i > self.processed_images:
                self.processed_images = i
                self.analyze_frame()
           
            self.ui.frame_slider.setValue(i)
            # to the last frame ,stop playing
            if i == self.total_images - 1 and self.is_play:
                self.play_btn_clicked()
            # time.sleep(0.1)
            cv2.waitKey(15)

    def merge_keypoint_datas(self,pred_instances):
        person_kpts = []
        for person in pred_instances:
            kpts = np.round(person['keypoints'][0],2)
            kpt_scores = np.round(person['keypoint_scores'][0],2)
            kpt_datas = np.hstack((kpts, kpt_scores.reshape(-1, 1)))
            person_kpts.append(kpt_datas)
        return person_kpts

    def merge_person_datas(self, frame_num, person_ids, person_bboxes, person_kpts):
        for pid, bbox, kpts in zip(person_ids, person_bboxes, person_kpts):
            new_kpts = np.zeros((len(self.kpts_dict),kpts.shape[1]))
            new_kpts[:17] = kpts
            new_kpts[17:, 2] = 0.9
            self.person_data.append({
                'frame_number': frame_num,
                'person_id': pid,
                'bbox': bbox,
                'keypoints': new_kpts
            })
        self.person_df = pd.DataFrame(self.person_data)

    def play_btn_clicked(self):
        if self.video_path == "":
            QMessageBox.warning(self, "無法開始播放", "請先讀取影片!")
            return
        self.is_play = not self.is_play
        if self.is_play:
            self.ui.play_btn.setText("Pause")
            self.play_frame(self.ui.frame_slider.value())
        else:
            self.ui.play_btn.setText("Play")

    def update_person_df(self):
        try:
            # 获取当前选择的人员 ID
            person_id = int(self.ui.ID_selector.currentText())
            # 获取当前帧数
            frame_num = self.ui.frame_slider.value()
            # 获取表格中的数据并更新到 DataFrame 中
            for kpt_idx in range(self.ui.Keypoint_Table.rowCount()):
                kpt_name = self.ui.Keypoint_Table.item(kpt_idx, 0).text()
                kpt_x = float(self.ui.Keypoint_Table.item(kpt_idx, 1).text())
                kpt_y = float(self.ui.Keypoint_Table.item(kpt_idx, 2).text())
                # 更新 DataFrame 中对应的值
                self.person_df.loc[(self.person_df['frame_number'] == frame_num) &
                                   (self.person_df['person_id'] == person_id), 'keypoints'].iloc[0][kpt_idx][:2] = [kpt_x, kpt_y]
            
            self.update_frame()
        except ValueError:
            print("无法将文本转换为数字")
        except IndexError:
            print("索引超出范围")

    def start_analyze_frame(self):
        for i in range(0, self.total_images-1):
            image = self.video_images[i]
            pred_instances, person_ids = process_one_image(self.args, image, self.detector, self.pose_estimator, self.tracker)
            # the keyscore > 1.0??
            person_kpts = self.merge_keypoint_datas(pred_instances)
            person_bboxes = pred_instances['bboxes']
            self.merge_person_datas(i, person_ids, person_bboxes, person_kpts)
            self.processed_frames.add(i)

    def analyze_frame(self):
        frame_num = self.ui.frame_slider.value()
        self.ui.frame_num_label.setText(
            f'{frame_num}/{self.total_images - 1}')

        # no image to analyze
        if self.total_images <= 0:
            return
        
        ori_image=self.video_images[frame_num].copy()
        image = ori_image.copy()

        if frame_num not in self.processed_frames:
            self.timer.tic()
            pred_instances, person_ids = process_one_image(self.args,image,self.detector,self.pose_estimator,self.tracker)
            average_time = self.timer.toc()
            fps= int(1/max(average_time,0.00001))
            self.ui.fps_label.setText(f"FPS: {fps}")
            # print(f"process_one_image FPS: {fps}")
            # pred_instances, person_ids = process_one_image(self.args,image,self.detector,self.pose_estimator,self.tracker)
            # the keyscore > 1.0??
            person_kpts = self.merge_keypoint_datas(pred_instances)
            person_bboxes = pred_instances['bboxes']
            self.merge_person_datas(frame_num, person_ids, person_bboxes, person_kpts)
            self.processed_frames.add(frame_num)
        
        if self.ui.frame_slider.value() == (self.total_images-1):
            self.ui.play_btn.click()

        if not self.ID_lock:
            self.import_id_to_selector(frame_num)
        else:
            self.check_id_exist(frame_num)
        self.import_data_to_table(frame_num)  
        
    def update_frame(self):
        
        curr_person_df, frame_num= self.obtain_curr_data()
        image=self.video_images[frame_num].copy()
        # if self.ui.show_skeleton_checkBox.isChecked():
        if not curr_person_df.empty:
            image = draw_points_and_skeleton(image, curr_person_df, joints_dict()['coco']['skeleton_links'], 
                                            points_color_palette='gist_rainbow', skeleton_palette_samples='jet',
                                            points_palette_samples=10, confidence_threshold=0.3)
            if self.ui.show_bbox_checkbox.isChecked():
                self.draw_bbox(curr_person_df, image)
        # 将原始图像直接显示在 QGraphicsView 中
        self.show_image(image, self.video_scene, self.ui.Frame_View)

    def obtain_curr_data(self):
        frame_num = self.ui.frame_slider.value()
        if self.ui.show_skeleton_checkBox.isChecked():
            try :
                if self.ui.ID_selector.count() > 0:
                    person_id = int(self.ui.ID_selector.currentText())
                    curr_person_df = self.person_df.loc[(self.person_df['frame_number'] == frame_num) &
                                                        (self.person_df['person_id'] == person_id)]
            except ValueError:
                print("valueError")
        else:
            curr_person_df = self.person_df.loc[(self.person_df['frame_number'] == frame_num)]
        return curr_person_df, frame_num

    def check_id_exist(self,frame_num):
        try:
            curr_person_df = self.person_df.loc[(self.person_df['frame_number'] == frame_num)]
            if curr_person_df.empty:
                print("check_id_exist未找到特定幀的數據")
                return
            person_ids = sorted(curr_person_df['person_id'].unique())
            if self.select_id not in person_ids:
                print("True")
                self.ui.ID_Locker.click()

        except KeyError:
            # print("未找到'frame_number'或'person_id'列")
            pass

    def import_id_to_selector(self, frame_num):
        try:
            self.ui.ID_selector.clear()
            filter_person_df = self.person_df.loc[(self.person_df['frame_number'] == frame_num)]
            if filter_person_df.empty:
                print("import_id_to_selector未找到特定幀的數據")
                return

            person_ids = sorted(filter_person_df['person_id'].unique())
            for person_id in person_ids:
                self.ui.ID_selector.addItem(str(person_id))

        except KeyError:
            # print("未找到'frame_number'或'person_id'列")
            pass

    def import_data_to_table(self, frame_num):
        try:
            person_id = self.ui.ID_selector.currentText()
            if person_id :
                person_id = int(person_id)
            else :
                return
            # 清空表格視圖
            self.clear_table_view()

            # 獲取特定人員在特定幀的數據
            person_data = self.person_df.loc[(self.person_df['frame_number'] == frame_num) & (self.person_df['person_id'] == person_id)]

            if person_data.empty:
                print("未找到特定人員在特定幀的數據")
                self.clear_table_view()
                return

            # 確保表格視圖大小足夠
            num_keypoints = len(self.kpts_dict)
            if self.ui.Keypoint_Table.rowCount() < num_keypoints:
                self.ui.Keypoint_Table.setRowCount(num_keypoints)

            # 將關鍵點數據匯入到表格視圖中
            for kpt_idx, kpt in enumerate(person_data['keypoints'].iloc[0]): 
                kptx, kpty = kpt[0], kpt[1]
                kpt_name = self.kpts_dict[kpt_idx]
                kpt_name_item = QTableWidgetItem(str(kpt_name))
                kptx_item = QTableWidgetItem(str(kptx))
                kpty_item = QTableWidgetItem(str(kpty))
                kpt_name_item.setTextAlignment(Qt.AlignRight)
                kptx_item.setTextAlignment(Qt.AlignRight)
                kpty_item.setTextAlignment(Qt.AlignRight)
                self.ui.Keypoint_Table.setItem(kpt_idx, 0, kpt_name_item)
                self.ui.Keypoint_Table.setItem(kpt_idx, 1, kptx_item)
                self.ui.Keypoint_Table.setItem(kpt_idx, 2, kpty_item)
            self.update_frame()
        except ValueError:
            print("未找到人員的ID列表")
        except AttributeError:
            print("未找到人員ID")
        except KeyError:
            print("人員ID在列表中不存在")
            
    def clear_table_view(self):
        self.ui.Keypoint_Table.clear()
        self.ui.Keypoint_Table.setColumnCount(3)
        title = ["Keypoint", "X", "Y"]
        self.ui.Keypoint_Table.setHorizontalHeaderLabels(title)

    def draw_bbox(self, person_data, img):
        person_ids = person_data['person_id']
        person_bbox = person_data['bbox']
        for id, bbox in zip(person_ids, person_bbox):
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            color = tuple(self.colors[id % len(self.colors)])
            img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
            img = cv2.putText(img, str(id), (x1, y1-10), cv2.FONT_HERSHEY_COMPLEX, 1.5, color, 2)
        return img

    def on_cell_clicked(self, row, column):
        self.correct_kpt_idx = row
        self.label_kpt = True
    
    def send_to_table(self, kptx, kpty):
    
        kptx_item = QTableWidgetItem(str(kptx))
        kpty_item = QTableWidgetItem(str(kpty))
        kptx_item.setTextAlignment(Qt.AlignRight)
        kpty_item.setTextAlignment(Qt.AlignRight)
        self.ui.Keypoint_Table.setItem(self.correct_kpt_idx, 1, kptx_item)
        self.ui.Keypoint_Table.setItem(self.correct_kpt_idx, 2, kpty_item)
        self.update_person_df()
            
    def mousePressEvent(self, event):
        if self.label_kpt:
            pos = event.pos()
            scene_pos = self.ui.Frame_View.mapToScene(pos)
            kptx, kpty = scene_pos.x(), scene_pos.y()
            if event.button() == Qt.LeftButton:
                self.send_to_table(kptx, kpty)
            elif event.button() == Qt.RightButton:
                print("t")
                kptx, kpty = 0, 0
                self.send_to_table(kptx, kpty)
            self.label_kpt = False

    def smooth_data(self):
        for curr_frame_num in self.processed_frames:
            person_id = int(self.ui.ID_selector.currentText())
            if curr_frame_num == 0:
                pre_frame_num = 0
            else:
                pre_frame_num = curr_frame_num - 1

            pre_person_data = self.person_df.loc[(self.person_df['frame_number'] == pre_frame_num) &
                                                (self.person_df['person_id'] == person_id)]
            curr_person_data = self.person_df.loc[(self.person_df['frame_number'] == curr_frame_num) &
                                                (self.person_df['person_id'] == person_id)]

            if not curr_person_data.empty:
                pre_kpts = pre_person_data.iloc[0]['keypoints']
                curr_kpts = curr_person_data.iloc[0]['keypoints']
                smoothed_kpts = []
                for pre_kpt, curr_kpt in zip(pre_kpts, curr_kpts): 
                    pre_kptx, pre_kpty = pre_kpt[0], pre_kpt[1]
                    curr_kptx , curr_kpty = curr_kpt[0], curr_kpt[1]
                    if pre_kptx != 0 and pre_kpty != 0 and curr_kptx != 0 and curr_kpty !=0:                 
                        curr_kptx = self.smooth_tool(curr_kptx, pre_kptx)
                        curr_kpty = self.smooth_tool(curr_kpty, pre_kpty)
                    else:
                        pass
                    smoothed_kpts.append([curr_kptx, curr_kpty, 0.9])  # 设置可信度为默认值
                # 更新 DataFrame 中的数据
                self.person_df.at[curr_person_data.index[0], 'keypoints'] = smoothed_kpts
            else:
                print("Previous frame data not found for smooth.")
        print("Smooth process is finished.")
  
    def ID_locker(self):
        if not self.ID_lock:
            self.ui.ID_selector.setEnabled(False)
            self.select_id = int(self.ui.ID_selector.currentText())
            self.ID_lock = True
            self.ui.ID_Locker.setText("Unlock")
        else:
            self.ui.ID_Locker.setText("Lock")
            self.select_id = 0
            self.ID_lock = False
            self.ui.ID_selector.setEnabled(True)

    def show_store_window(self):
        if self.person_df.empty:
            print("no data")
            return
        else:
            self.store_window = Store_Widget(self.video_name, self.video_images, self.person_df)
            self.store_window.show()
    
    def linear_interpolation(self):
        try:
            end = self.ui.frame_slider.value()
            start = end - 30
            
            # 获取当前帧的人员数据和上一帧的人员数据
            last_person_data = self.obtain_curr_data()[0]['keypoints'].iloc[0]
            person_id = int(self.ui.ID_selector.currentText())
            curr_person_data = self.person_df.loc[(self.person_df['frame_number'] == start) &
                                                (self.person_df['person_id'] == person_id)]
            # 确保当前帧和上一帧的数据都存在
            if not curr_person_data.empty:
                curr_person_data = curr_person_data.iloc[0]['keypoints']
                self.ui.frame_slider.setValue(start)
                # 计算每个关键点的线性插值的差值
                diff = np.subtract(last_person_data[17:], curr_person_data[17:])
                diff[:, 2] = 0.9
                # 对后续帧应用线性插值
                i = 1
                for frame_num in range(start, end):
                    for index, row in self.person_df[self.person_df['frame_number'] == frame_num].iterrows():
                        relative_distance = i / (end - start)  # 计算相对距离
                        self.person_df.at[index, 'keypoints'][17:] = np.add(curr_person_data[17:], relative_distance * diff)
                    i += 1
            else:
                print("Previous frame data not found for interpolation.")
        except UnboundLocalError:
            print("An error occurred in linear_interpolation function")
        except ValueError:
            print("ValueError occurred in linear_interpolation function")

    def keyPressEvent(self, event):
            if event.key() == ord('D') or event.key() == ord('d'):
                self.ui.frame_slider.setValue(self.ui.frame_slider.value() + 1)
            elif event.key() == ord('A') or event.key() == ord('a'):
                self.ui.frame_slider.setValue(self.ui.frame_slider.value() - 1)
            else:
                super().keyPressEvent(event)


    def load_json(self):
        # 打開文件對話框以選擇 JSON 文件
        json_path, _ = QFileDialog.getOpenFileName(
            self, "选择要加载的 JSON 文件", "", "JSON 文件 (*.json)")

        # 如果用戶取消選擇文件，則返回
        if not json_path:
            return

        try:
            # 从 JSON 文件中读取数据并转换为 DataFrame
            self.person_df = pd.read_json(json_path)
            process_frame_nums = sorted(self.person_df['frame_number'].unique())
            self.processed_frames = set(process_frame_nums) 
            self.import_id_to_selector(0)
            person_id = int(self.ui.ID_selector.currentText())

            person_kpt = self.person_df.loc[(self.person_df['person_id'] == person_id)]['keypoints']
            x_kpt_datas = []
            y_kpt_datas = []
            for kpt_datas in person_kpt:
                x_kpt_datas.append(kpt_datas[self.select_kpt_index][0])
                y_kpt_datas.append(kpt_datas[self.select_kpt_index][1])
            x_figure_dict = {'x':process_frame_nums, 'y': x_kpt_datas}
            y_figure_dict = {'x':process_frame_nums, 'y': y_kpt_datas}

            self.show_figure(x_figure_dict, self.ui.x_figure_view, 'X')
            self.show_figure(y_figure_dict, self.ui.y_figure_view, 'Y')
            self.update_frame()

        except Exception as e:
            print(f"加载 JSON 文件时出错：{e}")

    def show_figure(self, data, graphicview, title):
        scene = QGraphicsScene()
        pen = QPen(Qt.blue)
        scene.addText(title)

        xaxis = data['x']
        yaxis = data['y']

        # 設置圖表範圍
        x_min, x_max = min(xaxis), max(xaxis)
        y_min, y_max = min(yaxis), max(yaxis)

        # 計算轉換比例
        width = graphicview.width() 
        height = graphicview.height()
        x_scale = width / (x_max - x_min)
        y_scale = height / (y_max - y_min)

        prev_point = None
        for x, y in zip(xaxis, yaxis):
            x_coord = (x - x_min) * x_scale
            y_coord = height - (y - y_min) * y_scale  # flip y-axis
            if prev_point:
                scene.addLine(prev_point.x(), prev_point.y(), x_coord, y_coord, pen)
            prev_point = QPointF(x_coord, y_coord)

        graphicview.setScene(scene)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Pose_2d_Tab_Control()
    window.show()
    sys.exit(app.exec_())
