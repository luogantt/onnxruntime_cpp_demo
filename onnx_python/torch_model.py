import torch
import torchvision
import cv2
import onnx
import numpy as np
import timm
import os
from PIL import Image
from torchvision import transforms
import onnxruntime
from onnxsim import simplify

print(torch.__version__)
print(cv2.__version__)
print(np.__version__)
print(onnx.__version__)
classes = None
#class_file = r'F:\Pycharm\PyCharm_Study\Others\ONNX\Opencv-Onnx\classification_classes_ILSVRC2012.txt'

class_file = './classification_classes_ILSVRC2012.txt'
with open(class_file, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

def init_model(model_name):
    if model_name == 'alexnet':
        model = torchvision.models.alexnet(pretrained=True)
    if model_name == 'densnet':
        model = torchvision.models.densenet121(pretrained=True)
    if model_name == 'resnet':
        model = torchvision.models.resnet50(pretrained=True)
    if model_name == 'mobilenet':
        model = torchvision.models.mobilenet_v2(pretrained=True)
    if model_name == 'squeezenet':
        model = torchvision.models.squeezenet1_1(pretrained=True)
    if model_name == 'inception':
        model = torchvision.models.inception_v3(pretrained=False)
    if model_name == 'googlenet':
        model = torchvision.models.googlenet(pretrained=True)
    if model_name == 'vgg16':
        model = torchvision.models.vgg16(pretrained=True)
    if model_name == 'vgg19':
        model = torchvision.models.vgg19(pretrained=True)
    if model_name == 'shufflenet':
        model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
    if model_name == 'cspdarknet53':
        model = timm.create_model('cspdarknet53', pretrained=True)
    if model_name == 'seresnet18':
        model = timm.create_model('seresnet18', pretrained=True)
    if model_name == 'senet154':
        model = timm.create_model('senet154', pretrained=True)
    if model_name == 'seresnet50':
        model = timm.create_model('seresnet50', pretrained=True)
    if model_name == 'resnest50d':
        model = timm.create_model('resnest50d', pretrained=True)
    if model_name == 'skresnet50':
        model = timm.create_model('skresnet50', pretrained=True)
    model.eval()
    if model_name == 'inception':
        dummy = torch.randn(1, 3, 299, 299)
    else:
        dummy = torch.randn(1, 3, 224, 224)
    return model, dummy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model, dummy = init_model('vgg16')
model = model.to(device)
dummy = dummy.to(device)

#img_file = r'F:\Pycharm\PyCharm_Study\Others\ONNX\Opencv-Onnx\dog.jpg'

img_file = './dog.jpg'

################################torchvison????????????#####################################
transform = transforms.Compose([  # [1]
    transforms.Resize(256),  # [2]
    transforms.CenterCrop(224),  # [3]
    transforms.ToTensor(),  # [4]
    transforms.Normalize(  # [5]
        mean=[0.485, 0.456, 0.406],  # [6]
        std=[0.229, 0.224, 0.225]  # [7]
    )])

img = Image.open(img_file)
img_t = transform(img).to(device) # ????????????????????????????????????????????????
batch_t = torch.unsqueeze(img_t, 0)
tc_out = model(batch_t).detach().cpu().numpy()
# Get a class with a highest score.
tc_out = tc_out.flatten()
classId = np.argmax(tc_out)
confidence = tc_out[classId]

label = '%s: %.4f' % (classes[classId] if classes else 'Class #%d' % classId, confidence)



################################ pytorch???Onnx ######################################
onnx_name = 'vgg16.onnx'
torch.onnx.export(model, dummy, onnx_name)

# ????????????(?????????)
print("----- ???????????? -----")
# ?????????????????????????????????????????????
# ????????????
def check_onnx_output(filename, input_data, torch_output):
    session = onnxruntime.InferenceSession(filename)
    input_name = session.get_inputs()[0].name
    result = session.run([], {input_name: input_data.detach().cpu().numpy()})
    for test_result, gold_result in zip(result, torch_output.values()):
        np.testing.assert_almost_equal(
            gold_result.cpu().numpy(), test_result, decimal=3,
        )
    return result
# ????????????
def check_onnx_model(model, onnx_filename, input_image):
    with torch.no_grad():
        torch_out = {"output": model(input_image)}
    check_onnx_output(onnx_filename, input_image, torch_out)
    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)
    print("??????????????????")
    return onnx_model
# ???????????????onnx??????????????????
# ???????????????????????????????????????????????????????????????
onnx_model = check_onnx_model(model, onnx_name, dummy)

# -----  ????????????
print("-----  ???????????? -----")
# ??????onnx-simplifier???????????????https://github.com/daquexian/onnx-simplifier
# ????????????????????????python3 -m onnxsim input_onnx_model output_onnx_model
# ????????????????????????????????????https://convertmodel.com/

# ???????????????
filename = onnx_name + "sim.onnx"
# ????????????
# ??????skip_fuse_bn=True??????????????????bn??????pytorch???????????????bn????????????
simplified_model, check = simplify(onnx_model, skip_fuse_bn=True)
onnx.save_model(simplified_model, filename)
onnx.checker.check_model(simplified_model)
# ????????????
assert check, "??????????????????"
print("??????????????????")
################################ pytorch???Onnx ######################################



print(label)
################################ torchvison???????????? ######################################


################################ Opencv dnn ?????? Onnx #####################################
# ??????onnx??????
model_ = onnx.load(onnx_name)
# ??????IR????????????
onnx.checker.check_model(model_)
# opencv dnn??????
net = cv2.dnn.readNetFromONNX(onnx_name)

frame = cv2.imread(img_file)
# Create a 4D blob from a frame.
inpWidth = dummy.shape[-2]
inpHeight = dummy.shape[-2]
# blob = cv2.dnn.blobFromImage(frame, size=(inpWidth, inpHeight), crop=False)
blob = cv2.dnn.blobFromImage(frame,
                             scalefactor=1.0 / 255,
                             size=(inpWidth, inpHeight),
                             mean=[0.485, 0.456, 0.406],
                             swapRB=True,
                             crop=False)
# Run a model
net.setInput(blob)
out = net.forward()
print(out.shape)        # ?????????????????????

# Get a class with a highest score.
out = out.flatten()
classId = np.argmax(out)
confidence = out[classId]

# Put efficiency information.
t, _ = net.getPerfProfile()
label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
print(label)
cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

# Print predicted class.
label = '%s: %.4f' % (classes[classId] if classes else 'Class #%d' % classId, confidence)
print(label)
cv2.putText(frame, label, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
winName = 'onnx'

cv2.imshow(winName, frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
################################ Opencv dnn ?????? Onnx #####################################




