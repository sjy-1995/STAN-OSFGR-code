import scipy.io

# ############################### Step 1 ######################################

# data = scipy.io.loadmat('cars_annos.mat')
# class_names = data['class_names']
# f_class = open('./label_map.txt', 'w')
#
# num = 1
# for j in range(class_names.shape[1]):
#     class_name = str(class_names[0, j][0]).replace(' ', '_')
#     print(num, class_name)
#     f_class.write(str(num) + ' ' + class_name + '\n')
#     num = num + 1
# f_class.close()

# ################################ Step 2 ######################################

data = scipy.io.loadmat('cars_annos.mat')
annotations = data['annotations']
f_train = open('./mat2txt.txt', 'w')

num = 1
for i in range(annotations.shape[1]):
    name = str(annotations[0, i][0])[2:-2]
    test = int(annotations[0, i][6])
    clas = int(annotations[0, i][5])

    name = str(name)
    clas = str(clas)
    test = str(test)
    f_train.write(str(num) + ' ' + name + ' ' + clas + ' ' + test + '\n')
    num = num + 1

f_train.close()



