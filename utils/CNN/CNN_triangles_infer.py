import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# identyczny model jak w treningu
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3,16,3,padding=1)
        self.c2 = nn.Conv2d(16,32,3,padding=1)
        self.c3 = nn.Conv2d(32,64,3,padding=1)
        self.fc1 = nn.Linear(64*4*4,64)
        self.fc2 = nn.Linear(64,2)  # 0=not_tri, 1=tri
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.c1(x)),2)
        x = F.max_pool2d(F.relu(self.c2(x)),2)
        x = F.max_pool2d(F.relu(self.c3(x)),2)
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def load_model(ckpt_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    model = TinyCNN().to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    return model, device

_tf = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor()])

def predict_is_triangle(model, device, pil_image, thresh=0.5):
    x = _tf(pil_image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        p = F.softmax(model(x), dim=1)[0,1].item()
    return (p >= thresh), p  # (bool, confidence)

# przykład użycia:
if __name__ == "__main__":
    model, device = load_model(r"E:\BOT ANK\bot\moje_AI\yolov8\FULL BOT\tri_cnn.pt")
    img = Image.open(r"E:\ścieżka\do\cropa.png")  # 32x32 lub większy
    is_tri, conf = predict_is_triangle(model, device, img, thresh=0.6)
    print("triangle?" , is_tri, "conf=", round(conf,3))
