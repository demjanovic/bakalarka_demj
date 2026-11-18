from pathlib import Path

usb = Path("E:/")

folders = [
    "Features_For_BT",
    "Features_For_BT1",
    "Features_For_BT2",
    "Features_For_BT3"
]

for folder in folders:
    folder_path = usb / folder

    print(f"\nPriecinok: {folder}")

    if not folder_path.exists():
        print("POZOR: Priecinok sa nenasiel!")
        continue

    files = list(folder_path.rglob("*.pt"))
    print(f"Pocet .pt suborov: {len(files)}")
    print("Prve subory:")

    for file in files[:3]:
        print(" ", file.name)