
import math

def rotatePoint(pointA: tuple[int, int], pointB: tuple[int, int], angle: float) -> tuple[int, int]:
  # 将角度从度转换为弧度
  angleRad = (angle * math.pi) / 180

  # 移动坐标系使旋转中心在(0,0)
  x = pointB[0] - pointA[0]
  y = pointB[1] - pointA[1]

  # 应用旋转公式
  newX = x * math.cos(angleRad) - y * math.sin(angleRad)
  newY = x * math.sin(angleRad) + y * math.cos(angleRad)

  # 将坐标系移回原位
  newX += pointA[0]
  newY += pointA[1]

  return (math.floor(newX), math.floor(newY))

