for file in *.FCStd; do
  freecad --console SketchToTemplate.FCMacro "$file"
done
