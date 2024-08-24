import tkinter as tk
import module.AugRotation as ar
import importlib
importlib.reload(ar)


aug = ar.AugRotation()  

def click():
    img_folder = text.get()
    label_folder = img_folder.replace("Fallen_Images", "Converted_Labels")
    aug.view_sample_box_image(img_folder, label_folder, 4)

window=tk.Tk()

window.title("랜덤하게 이미지 뽑기")

window.geometry("1000x1000+100+100")
window.resizable(True, True)

label=tk.Label(window,text="이미지 주소를 입력하시오",width=30,height=3)
label.pack()

text=tk.Entry(window)
text.insert(tk.END, "")
text.pack()

button=tk.Button(window,text="이미지 생성",width=10,height=3, command=click)
button.pack()

window.mainloop()