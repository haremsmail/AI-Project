# 🚀 GETTING STARTED - 5 MINUTES

Follow these steps to run your Mango Leaf Disease Classifier!

---

## ✅ STEP 1: Open PowerShell

1. Press `Windows Key + R`
2. Type: `powershell`
3. Press `Enter`

---

## ✅ STEP 2: Navigate to Project

Copy and paste this:

```powershell
cd c:\Desktop\week1\AI-Project\finalproject
```

Press `Enter`

---

## ✅ STEP 3: Activate Virtual Environment

Copy and paste this:

```powershell
.\venv\Scripts\Activate.ps1
```

Press `Enter`

You should see `(venv)` appear at the beginning of the line.

---

## ✅ STEP 4: RUN THE PROGRAM

Copy and paste this:

```powershell
python main.py
```

Press `Enter`

---

## 🎯 WHAT TO DO NEXT

A menu will appear:

```
🥭 MANGO LEAF DISEASE CLASSIFIER
==================================================

1 - Train all models
2 - Test on single image  
3 - Compare model accuracy
4 - Exit

Select option (1-4):
```

### **First Time? DO THIS:**
1. Type: `1` and press `Enter`
2. Models will train automatically (takes 5-10 minutes)
3. Wait for: `✓ All models trained and saved!`
4. You're done! The models are now ready to use.

### **After Training? TRY THIS:**
- Type: `3` to see which model is best
- Type: `2` to test on your own image
- Type: `4` to exit

---

## 📸 TESTING WITH AN IMAGE

When you select option 2:

```
🧪 TEST SINGLE IMAGE
Select option (1-4): 2

Enter image path: 
```

You need to provide the full path to an image file, for example:

```
C:\Users\YourName\Downloads\mango_leaf.jpg
```

Then press `Enter` and you'll see:

```
PREDICTION RESULT
  Disease/Status: Powdery Mildew
  Confidence: 94%
  Best Model Used: SVM
```

---

## ⚠️ COMMON ISSUES

### Issue: Permission Denied
**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

Then try again.

### Issue: "venv not found"
**Solution:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Issue: "Dataset not found"
**Solution:**
```powershell
python setup_dataset.py
```

### Issue: Model takes forever
**Solution:** It's normal! First time training takes 5-10 minutes. Grab a coffee ☕

---

## 📋 COMPLETE ONE-LINER (Copy & Paste)

If you want to do everything at once:

```powershell
cd c:\Desktop\week1\AI-Project\finalproject; .\venv\Scripts\Activate.ps1; python main.py
```

---

## ✨ YOU'RE ALL SET!

Your Mango Leaf Disease Classifier is ready to use! 

Just remember:
1. Run `python main.py`
2. Select an option from the menu
3. Follow the prompts
4. See results!

---

**Need Help?** Read `SIMPLE_README.md` for more details.

**Happy classifying! 🥭**
