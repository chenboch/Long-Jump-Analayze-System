import sys
from PyQt5 import QtWidgets
from Widget.store_ui import store_ui_widget
import os
import cv2
import shutil


class Store_Widget(QtWidgets.QWidget):
    def __init__(self, video_name, video_images, person_df):
        super().__init__()
        self.ui = store_ui_widget()
        self.ui.setupUi(self)
        
        # 连接按钮的信号和槽
        self.ui.store_btn.clicked.connect(self.store)
        self.ui.cancel_btn.clicked.connect(self.cancel)
        self.person_df = person_df
        self.video_name = video_name
        self.video_images = video_images
        self.select_id = []
        self.add_id_checkbox()

    def store(self):
        self.save_datas()
        print("Store data success")
        self.close()

    def add_id_checkbox(self):
        person_ids = sorted(self.person_df['person_id'].unique())
        for person_id in person_ids:
            checkbox = QtWidgets.QCheckBox(f"{person_id}")
            checkbox.clicked.connect(lambda state, chk=checkbox: self.add_id_to_select(chk))
            self.ui.dispaly_id_layout.addWidget(checkbox)

    def add_id_to_select(self, checkbox):
        if checkbox.isChecked():
            # print(f"Checkbox with label '{checkbox.text()}' is checked.")
            self.select_id.append(int(checkbox.text()))
        else:
            self.select_id.remove(int(checkbox.text()))

    def reset_person_df(self):
        def reset_keypoints(keypoints):
            modified_keypoints = keypoints.copy()
            for kpt_idx, kpt in enumerate(keypoints):
                kptx, kpty = kpt[0], kpt[1]
                if kptx == 0.0 and kpty == 0.0:
                    modified_keypoints[kpt_idx][2] = 0
            return modified_keypoints

        self.person_df['keypoints'] = self.person_df['keypoints'].apply(reset_keypoints)

    def filter_person_df(self):
        filter_df = self.person_df[self.person_df['person_id'].isin(self.select_id)]
        return filter_df
    
    def save_datas(self):
        output_folder = f"../../Db/output/{self.video_name}"
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)    
        os.makedirs(output_folder)
        
        # save frame image            
        for i, frame in enumerate(self.video_images):
            frame_path = os.path.join(output_folder, f"{i}.jpg")
            cv2.imwrite(frame_path, frame)

       # 将 DataFrame 保存为 JSON 文件
        json_path = os.path.join(output_folder, f"{self.video_name}.json")
        self.reset_person_df()

        if not self.select_id:
            save_person_df = self.person_df
        else:
            save_person_df = self.filter_person_df()
        save_person_df.to_json(json_path, orient='records')

    def cancel(self):
        self.close()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    widget = Store_Widget()
    widget.show()
    sys.exit(app.exec_())
