import cv2
import numpy as np

digits = cv2.imread("digits.png", cv2.IMREAD_GRAYSCALE)
test_digits = cv2.imread("test_digits.png", cv2.IMREAD_GRAYSCALE)

rows = np.vsplit(digits,50)
#cv2.imshow("row0",rows[0])

cells = []
for row in rows:
    row_cells = np.hsplit(row,50)
#    cv2.imshow("row0",row_cells[0])
    for cell in row_cells:
        cell = cell.flatten()
        cells.append(cell)
cells = np.array(cells, dtype = np.float32)
        
k = np.arange(10)
cell_labels = np.repeat(k,250)

test_cells = np.vsplit(test_digits,50)
test_cells_f = []
for td in test_cells:
    td = td.flatten()
    test_cells_f.append(td)
test_cells_f = np.array(test_cells_f, dtype = np.float32)
#cv2.imshow("test",test_cells[0])

#KNN
knn = cv2.ml.KNearest_create()
knn.train(cells,cv2.ml.ROW_SAMPLE, cell_labels)
ret, result, neighbours, dist = knn.findNearest(test_cells_f, k=1)
print(result)

#cv2.imshow("Digits", digits)
cv2.waitKey()
cv2.destroyAllWindows()
