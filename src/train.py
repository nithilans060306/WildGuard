from ultralytics import RTDETR

model = RTDETR(r"C:\WildGuard\models\wildguard\weights\best.pt")

if __name__ == '__main__':
    model.train(
        data=r"C:\WildGuard\configs\wildguard.yaml",
        epochs=45,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        project=r"C:\WildGuard\models",
        name="wildguard",
        exist_ok=True,
        patience=10,
        resume=True
    )