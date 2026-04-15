param(
    [Parameter(Mandatory = $true)]
    [string]$InputJson,
    [Parameter(Mandatory = $true)]
    [string]$MainFigure,
    [Parameter(Mandatory = $true)]
    [string]$DecompositionFigure,
    [Parameter(Mandatory = $true)]
    [string]$EpsilonFigure
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Drawing

function Get-ColorFromHex {
    param([string]$Hex)

    $value = $Hex.TrimStart("#")
    return [System.Drawing.Color]::FromArgb(
        [Convert]::ToInt32($value.Substring(0, 2), 16),
        [Convert]::ToInt32($value.Substring(2, 2), 16),
        [Convert]::ToInt32($value.Substring(4, 2), 16)
    )
}

function Save-Bitmap {
    param(
        [System.Drawing.Bitmap]$Bitmap,
        [string]$Path
    )

    $directory = Split-Path -Parent $Path
    if ($directory) {
        New-Item -ItemType Directory -Force -Path $directory | Out-Null
    }
    $Bitmap.Save($Path, [System.Drawing.Imaging.ImageFormat]::Png)
}

function Get-LinearX {
    param(
        [double]$Value,
        [double]$Min,
        [double]$Max,
        [double]$Left,
        [double]$Width
    )

    if ($Max -le $Min) {
        return [float]$Left
    }
    return [float]($Left + (($Value - $Min) / ($Max - $Min)) * $Width)
}

function Get-LogX {
    param(
        [double]$Value,
        [double]$Min,
        [double]$Max,
        [double]$Left,
        [double]$Width
    )

    $logMin = [Math]::Log10($Min)
    $logMax = [Math]::Log10($Max)
    $logValue = [Math]::Log10($Value)
    return [float]($Left + (($logValue - $logMin) / ($logMax - $logMin)) * $Width)
}

function Get-Y {
    param(
        [double]$Value,
        [double]$Min,
        [double]$Max,
        [double]$Top,
        [double]$Height
    )

    if ($Max -le $Min) {
        return [float]($Top + $Height)
    }
    return [float]($Top + $Height - (($Value - $Min) / ($Max - $Min)) * $Height)
}

function Draw-CenteredString {
    param(
        [System.Drawing.Graphics]$Graphics,
        [string]$Text,
        [System.Drawing.Font]$Font,
        [System.Drawing.Brush]$Brush,
        [float]$CenterX,
        [float]$Y
    )

    $size = $Graphics.MeasureString($Text, $Font)
    $Graphics.DrawString($Text, $Font, $Brush, $CenterX - ($size.Width / 2), $Y)
}

function Draw-RotatedString {
    param(
        [System.Drawing.Graphics]$Graphics,
        [string]$Text,
        [System.Drawing.Font]$Font,
        [System.Drawing.Brush]$Brush,
        [float]$CenterX,
        [float]$CenterY,
        [float]$Angle
    )

    $state = $Graphics.Save()
    $Graphics.TranslateTransform($CenterX, $CenterY)
    $Graphics.RotateTransform($Angle)
    $size = $Graphics.MeasureString($Text, $Font)
    $Graphics.DrawString($Text, $Font, $Brush, -($size.Width / 2), -($size.Height / 2))
    $Graphics.Restore($state)
}

function Get-OrderedSubset {
    param(
        [object[]]$Rows,
        [string]$Dataset,
        [string[]]$FamilyOrder
    )

    $subset = @()
    foreach ($family in $FamilyOrder) {
        $row = $Rows | Where-Object { $_.dataset -eq $Dataset -and $_.family -eq $family }
        if ($row) {
            $subset += $row
        }
    }
    return $subset
}

function New-GraphicsContext {
    param(
        [int]$Width,
        [int]$Height
    )

    $bitmap = New-Object System.Drawing.Bitmap $Width, $Height
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.Clear([System.Drawing.Color]::White)
    $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit
    return @{ Bitmap = $bitmap; Graphics = $graphics }
}

$rows = Get-Content -Raw $InputJson | ConvertFrom-Json
$datasetOrder = @("cifar10", "mufac")
$datasetDisplay = @{
    cifar10 = "CIFAR-10"
    mufac = "MUFAC"
}
$familyOrder = @("baseline_train", "MSG", "FanchuanUnlearning", "SCRUB", "CT", "DELETE")
$familyDisplay = @{
    baseline_train = "Train"
    MSG = "MSG"
    FanchuanUnlearning = "Fanchuan"
    SCRUB = "SCRUB"
    CT = "CT"
    DELETE = "DELETE"
}
$familyColors = @{
    baseline_train = (Get-ColorFromHex "#6c757d")
    MSG = (Get-ColorFromHex "#1b9e77")
    FanchuanUnlearning = (Get-ColorFromHex "#377eb8")
    SCRUB = (Get-ColorFromHex "#d95f02")
    CT = (Get-ColorFromHex "#7570b3")
    DELETE = (Get-ColorFromHex "#e7298a")
}
$scoreColors = @{
    forgetting_quality = (Get-ColorFromHex "#4c78a8")
    retain_ratio = (Get-ColorFromHex "#59a14f")
    test_ratio = (Get-ColorFromHex "#e15759")
}
$epsilonColors = @{
    low = (Get-ColorFromHex "#2b8cbe")
    high = (Get-ColorFromHex "#de2d26")
}
$labelOffsets = @{
    baseline_train = @(6, -18)
    MSG = @(6, 6)
    FanchuanUnlearning = @(6, 6)
    SCRUB = @(6, 6)
    CT = @(6, -18)
    DELETE = @(6, 6)
}

$axisColor = Get-ColorFromHex "#222222"
$gridColor = Get-ColorFromHex "#dddddd"
$textColor = Get-ColorFromHex "#222222"

$titleFont = New-Object System.Drawing.Font("Arial", 14, [System.Drawing.FontStyle]::Bold)
$panelTitleFont = New-Object System.Drawing.Font("Arial", 12, [System.Drawing.FontStyle]::Bold)
$axisFont = New-Object System.Drawing.Font("Arial", 9)
$labelFont = New-Object System.Drawing.Font("Arial", 9)
$legendFont = New-Object System.Drawing.Font("Arial", 9)
$brushText = New-Object System.Drawing.SolidBrush $textColor
$brushWhite = [System.Drawing.Brushes]::White
$penAxis = New-Object System.Drawing.Pen $axisColor, 1.4
$penGrid = New-Object System.Drawing.Pen $gridColor, 1.0
$penDashed = New-Object System.Drawing.Pen (Get-ColorFromHex "#555555"), 1.4
$penDashed.DashStyle = [System.Drawing.Drawing2D.DashStyle]::Dash

$scatterCtx = New-GraphicsContext -Width 1380 -Height 520
$scatterBitmap = $scatterCtx.Bitmap
$scatterGraphics = $scatterCtx.Graphics
Draw-CenteredString -Graphics $scatterGraphics -Text "Benchmark score versus runtime tradeoff" -Font $titleFont -Brush $brushText -CenterX 690 -Y 10

$legendY = 40
$scatterGraphics.FillEllipse((New-Object System.Drawing.SolidBrush $familyColors.MSG), 375, $legendY, 10, 10)
$scatterGraphics.DrawEllipse((New-Object System.Drawing.Pen $familyColors.MSG, 1.8), 375, $legendY, 10, 10)
$scatterGraphics.DrawString("passes efficiency cutoff", $legendFont, $brushText, 390, $legendY - 2)
$scatterGraphics.FillEllipse($brushWhite, 610, $legendY, 10, 10)
$scatterGraphics.DrawEllipse((New-Object System.Drawing.Pen $familyColors.baseline_train, 1.8), 610, $legendY, 10, 10)
$scatterGraphics.DrawString("fails efficiency cutoff", $legendFont, $brushText, 625, $legendY - 2)
$scatterGraphics.DrawLine($penDashed, 830, $legendY + 5, 865, $legendY + 5)
$scatterGraphics.DrawString("5x efficiency cutoff", $legendFont, $brushText, 875, $legendY - 2)

for ($panelIndex = 0; $panelIndex -lt $datasetOrder.Count; $panelIndex++) {
    $dataset = $datasetOrder[$panelIndex]
    $subset = Get-OrderedSubset -Rows $rows -Dataset $dataset -FamilyOrder $familyOrder

    $panelX = 70 + ($panelIndex * 640)
    $panelY = 90
    $plotLeft = $panelX + 60
    $plotTop = $panelY + 35
    $plotWidth = 500
    $plotHeight = 300

    Draw-CenteredString -Graphics $scatterGraphics -Text $datasetDisplay[$dataset] -Font $panelTitleFont -Brush $brushText -CenterX ($panelX + 310) -Y $panelY

    foreach ($tick in @(0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)) {
        $y = Get-Y -Value $tick -Min 0.0 -Max 0.30 -Top $plotTop -Height $plotHeight
        $scatterGraphics.DrawLine($penGrid, $plotLeft, $y, $plotLeft + $plotWidth, $y)
        $scatterGraphics.DrawString(($tick.ToString("0.00")), $axisFont, $brushText, $plotLeft - 45, $y - 7)
    }

    foreach ($tick in @(1, 2, 5, 10, 20, 50)) {
        $x = Get-LogX -Value $tick -Min 0.8 -Max 60.0 -Left $plotLeft -Width $plotWidth
        $scatterGraphics.DrawLine($penGrid, $x, $plotTop, $x, $plotTop + $plotHeight)
        $scatterGraphics.DrawString(($tick.ToString("0")), $axisFont, $brushText, $x - 7, $plotTop + $plotHeight + 8)
    }

    $cutoffX = Get-LogX -Value 5.0 -Min 0.8 -Max 60.0 -Left $plotLeft -Width $plotWidth
    $scatterGraphics.DrawLine($penDashed, $cutoffX, $plotTop, $cutoffX, $plotTop + $plotHeight)
    $scatterGraphics.DrawLine($penAxis, $plotLeft, $plotTop, $plotLeft, $plotTop + $plotHeight)
    $scatterGraphics.DrawLine($penAxis, $plotLeft, $plotTop + $plotHeight, $plotLeft + $plotWidth, $plotTop + $plotHeight)
    Draw-CenteredString -Graphics $scatterGraphics -Text "Speedup vs. retraining" -Font $axisFont -Brush $brushText -CenterX ($plotLeft + ($plotWidth / 2)) -Y ($plotTop + $plotHeight + 30)

    if ($panelIndex -eq 0) {
        Draw-RotatedString -Graphics $scatterGraphics -Text "Raw benchmark score" -Font $axisFont -Brush $brushText -CenterX 25 -CenterY ($plotTop + ($plotHeight / 2)) -Angle -90
    }

    foreach ($row in $subset) {
        $x = Get-LogX -Value ([double]$row.speedup_vs_retrain) -Min 0.8 -Max 60.0 -Left $plotLeft -Width $plotWidth
        $y = Get-Y -Value ([double]$row.raw_final_score) -Min 0.0 -Max 0.30 -Top $plotTop -Height $plotHeight
        $radius = 6
        $pen = New-Object System.Drawing.Pen $familyColors[$row.family], 1.8
        if ([bool]$row.passed_efficiency_cutoff) {
            $brush = New-Object System.Drawing.SolidBrush $familyColors[$row.family]
        }
        else {
            $brush = $brushWhite
        }
        $scatterGraphics.FillEllipse($brush, $x - $radius, $y - $radius, 2 * $radius, 2 * $radius)
        $scatterGraphics.DrawEllipse($pen, $x - $radius, $y - $radius, 2 * $radius, 2 * $radius)
        $offset = $labelOffsets[$row.family]
        $scatterGraphics.DrawString($familyDisplay[$row.family], $labelFont, $brushText, $x + $offset[0], $y + $offset[1])
    }
}

Save-Bitmap -Bitmap $scatterBitmap -Path $MainFigure
$scatterGraphics.Dispose()
$scatterBitmap.Dispose()

$decompCtx = New-GraphicsContext -Width 1380 -Height 520
$decompBitmap = $decompCtx.Bitmap
$decompGraphics = $decompCtx.Graphics
Draw-CenteredString -Graphics $decompGraphics -Text "Score decomposition by dataset" -Font $titleFont -Brush $brushText -CenterX 690 -Y 10

$legendY = 40
$legendItems = @(
    @{ Label = "Forgetting quality"; Color = $scoreColors.forgetting_quality; X = 340 }
    @{ Label = "Retain ratio"; Color = $scoreColors.retain_ratio; X = 560 }
    @{ Label = "Test ratio"; Color = $scoreColors.test_ratio; X = 730 }
)
foreach ($item in $legendItems) {
    $brush = New-Object System.Drawing.SolidBrush $item.Color
    $decompGraphics.FillRectangle($brush, $item.X, $legendY, 12, 12)
    $decompGraphics.DrawString($item.Label, $legendFont, $brushText, $item.X + 18, $legendY - 1)
}

for ($panelIndex = 0; $panelIndex -lt $datasetOrder.Count; $panelIndex++) {
    $dataset = $datasetOrder[$panelIndex]
    $subset = Get-OrderedSubset -Rows $rows -Dataset $dataset -FamilyOrder $familyOrder

    $panelX = 70 + ($panelIndex * 640)
    $panelY = 90
    $plotLeft = $panelX + 60
    $plotTop = $panelY + 35
    $plotWidth = 500
    $plotHeight = 300

    Draw-CenteredString -Graphics $decompGraphics -Text $datasetDisplay[$dataset] -Font $panelTitleFont -Brush $brushText -CenterX ($panelX + 310) -Y $panelY

    foreach ($tick in @(0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2)) {
        $y = Get-Y -Value $tick -Min 0.0 -Max 1.2 -Top $plotTop -Height $plotHeight
        $decompGraphics.DrawLine($penGrid, $plotLeft, $y, $plotLeft + $plotWidth, $y)
        $decompGraphics.DrawString(($tick.ToString("0.0")), $axisFont, $brushText, $plotLeft - 35, $y - 7)
    }

    $decompGraphics.DrawLine($penAxis, $plotLeft, $plotTop, $plotLeft, $plotTop + $plotHeight)
    $decompGraphics.DrawLine($penAxis, $plotLeft, $plotTop + $plotHeight, $plotLeft + $plotWidth, $plotTop + $plotHeight)

    if ($panelIndex -eq 0) {
        Draw-RotatedString -Graphics $decompGraphics -Text "Score component value" -Font $axisFont -Brush $brushText -CenterX 25 -CenterY ($plotTop + ($plotHeight / 2)) -Angle -90
    }

    $groupSpacing = $plotWidth / $subset.Count
    $groupWidth = $groupSpacing * 0.72
    $barWidth = $groupWidth / 3

    for ($i = 0; $i -lt $subset.Count; $i++) {
        $row = $subset[$i]
        $groupLeft = $plotLeft + ($i * $groupSpacing) + (($groupSpacing - $groupWidth) / 2)
        $metrics = @(
            @{ Value = [double]$row.forgetting_quality; Color = $scoreColors.forgetting_quality }
            @{ Value = [double]$row.retain_ratio; Color = $scoreColors.retain_ratio }
            @{ Value = [double]$row.test_ratio; Color = $scoreColors.test_ratio }
        )

        for ($j = 0; $j -lt $metrics.Count; $j++) {
            $metric = $metrics[$j]
            $x = $groupLeft + ($j * $barWidth)
            $y = Get-Y -Value $metric.Value -Min 0.0 -Max 1.2 -Top $plotTop -Height $plotHeight
            $height = ($plotTop + $plotHeight) - $y
            $brush = New-Object System.Drawing.SolidBrush $metric.Color
            $decompGraphics.FillRectangle($brush, $x, $y, $barWidth - 2, $height)
            $decompGraphics.DrawRectangle($penAxis, $x, $y, $barWidth - 2, $height)
        }

        Draw-CenteredString -Graphics $decompGraphics -Text $familyDisplay[$row.family] -Font $axisFont -Brush $brushText -CenterX ($groupLeft + ($groupWidth / 2)) -Y ($plotTop + $plotHeight + 8)
    }
}

Save-Bitmap -Bitmap $decompBitmap -Path $DecompositionFigure
$decompGraphics.Dispose()
$decompBitmap.Dispose()

$epsilonCtx = New-GraphicsContext -Width 1380 -Height 520
$epsilonBitmap = $epsilonCtx.Bitmap
$epsilonGraphics = $epsilonCtx.Graphics
Draw-CenteredString -Graphics $epsilonGraphics -Text "Per-example epsilon category shares" -Font $titleFont -Brush $brushText -CenterX 690 -Y 10

$legendY = 40
$epsilonGraphics.FillRectangle((New-Object System.Drawing.SolidBrush $epsilonColors.low), 395, $legendY, 12, 12)
$epsilonGraphics.DrawString("epsilon ~= 0.693", $legendFont, $brushText, 415, $legendY - 1)
$epsilonGraphics.FillRectangle((New-Object System.Drawing.SolidBrush $epsilonColors.high), 635, $legendY, 12, 12)
$epsilonGraphics.DrawString("epsilon = 50", $legendFont, $brushText, 655, $legendY - 1)

for ($panelIndex = 0; $panelIndex -lt $datasetOrder.Count; $panelIndex++) {
    $dataset = $datasetOrder[$panelIndex]
    $subset = Get-OrderedSubset -Rows $rows -Dataset $dataset -FamilyOrder $familyOrder

    $panelX = 70 + ($panelIndex * 640)
    $panelY = 90
    $plotLeft = $panelX + 60
    $plotTop = $panelY + 35
    $plotWidth = 500
    $plotHeight = 300

    Draw-CenteredString -Graphics $epsilonGraphics -Text $datasetDisplay[$dataset] -Font $panelTitleFont -Brush $brushText -CenterX ($panelX + 310) -Y $panelY

    foreach ($tick in @(0.0, 0.25, 0.50, 0.75, 1.0)) {
        $y = Get-Y -Value $tick -Min 0.0 -Max 1.0 -Top $plotTop -Height $plotHeight
        $epsilonGraphics.DrawLine($penGrid, $plotLeft, $y, $plotLeft + $plotWidth, $y)
        $epsilonGraphics.DrawString((("{0:P0}" -f $tick)), $axisFont, $brushText, $plotLeft - 40, $y - 7)
    }

    $epsilonGraphics.DrawLine($penAxis, $plotLeft, $plotTop, $plotLeft, $plotTop + $plotHeight)
    $epsilonGraphics.DrawLine($penAxis, $plotLeft, $plotTop + $plotHeight, $plotLeft + $plotWidth, $plotTop + $plotHeight)

    if ($panelIndex -eq 0) {
        Draw-RotatedString -Graphics $epsilonGraphics -Text "Share of forget examples" -Font $axisFont -Brush $brushText -CenterX 25 -CenterY ($plotTop + ($plotHeight / 2)) -Angle -90
    }

    $groupSpacing = $plotWidth / $subset.Count
    $barWidth = $groupSpacing * 0.55

    for ($i = 0; $i -lt $subset.Count; $i++) {
        $row = $subset[$i]
        $barLeft = $plotLeft + ($i * $groupSpacing) + (($groupSpacing - $barWidth) / 2)
        $lowY = Get-Y -Value ([double]$row.epsilon_0p693_share) -Min 0.0 -Max 1.0 -Top $plotTop -Height $plotHeight
        $lowHeight = ($plotTop + $plotHeight) - $lowY
        $topY = Get-Y -Value 1.0 -Min 0.0 -Max 1.0 -Top $plotTop -Height $plotHeight
        $highY = $topY
        $highHeight = $lowY - $topY

        $epsilonGraphics.FillRectangle((New-Object System.Drawing.SolidBrush $epsilonColors.low), $barLeft, $lowY, $barWidth, $lowHeight)
        $epsilonGraphics.FillRectangle((New-Object System.Drawing.SolidBrush $epsilonColors.high), $barLeft, $highY, $barWidth, $highHeight)
        $epsilonGraphics.DrawRectangle($penAxis, $barLeft, $highY, $barWidth, $lowHeight + $highHeight)
        Draw-CenteredString -Graphics $epsilonGraphics -Text $familyDisplay[$row.family] -Font $axisFont -Brush $brushText -CenterX ($barLeft + ($barWidth / 2)) -Y ($plotTop + $plotHeight + 8)
    }
}

Save-Bitmap -Bitmap $epsilonBitmap -Path $EpsilonFigure
$epsilonGraphics.Dispose()
$epsilonBitmap.Dispose()

Write-Host "Saved main figure to $MainFigure"
Write-Host "Saved supporting figure to $DecompositionFigure"
Write-Host "Saved supporting figure to $EpsilonFigure"
