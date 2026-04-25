# cn-strategies

Code Night algoritmik trading yarışması strateji dosyaları.

## Katılım

1. Bu repoyu fork'la
2. `strategies/` altına takım adında bir klasör aç
3. Dosyalarını içine koy
4. Pull request aç

## Klasör Yapısı

```
strategies/
  takim-adiniz/
    strategy.py        # zorunlu
    model.pkl          # opsiyonel — eğitilmiş model dosyası
    (diğer dosyalar)   # opsiyonel — strategy.py'nin ihtiyaç duyduğu her şey
```

`strategy.py` zorunlu, geri kalanlar isteğe bağlı. Model veya yardımcı dosya kullanıyorsan `strategy.py` ile aynı klasöre koy.
