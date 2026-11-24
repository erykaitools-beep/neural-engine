# Neural Engine

Czy mała sieć może nauczyć się nowej reguły, nie tracąc poprzedniej? Neural Engine to projekt badawczy, w którym sprawdzamy to na symulowanych neuronach. Komórki reagują impulsami, przechowują krótkotrwały stan, a w doświadczeniach z uczeniem mogą zmieniać siłę swoich połączeń. Długoterminowym celem jest przetwarzanie sekwencji, a później języka. Obecne prototypy nie są modelem językowym ani dokładną symulacją ludzkiego mózgu.

## Co pokazały dotychczasowe próby?

- **Połączenia w 3D.** W sieci 16 neuronów położenie komórek decydowało o tym, które z nich wzajemnie się hamowały. Połączenia były geometrycznie krótsze, a aktywność komórek się zmieniła. Jednak w prostym zadaniu A→B, w czterech sparowanych uruchomieniach, końcowe decyzje były takie same jak przy losowym doborze sąsiadów.
- **Kiedy odpowiedzieć?** Ręcznie zaprojektowana bramka pamiętała poprzedni poziom sygnału. Odrzuciła 768 na 768 zakazów jednego rodzaju. Kiedy jednak zakaz zaczynał się tak samo jak zezwolenie, odpowiedziała za wcześnie we wszystkich 768 próbach. W tych testach sieć nie uczyła się reguły rytmu.
- **Kiedy zmienić wspomnienie?** W jednej kontrolowanej próbie wymaganie drugiego potwierdzenia przed osłabieniem połączenia ochroniło znaną odpowiedź przed jedną błędną korektą. Gdy reguła rzeczywiście się zmieniła, adaptacja wymagała jednej lekcji więcej niż przy dotychczasowym filtrze.

To wyniki odrębnych prototypów na wybranych zadaniach, nie dowód ogólnej zdolności ciągłego uczenia się. W tych doświadczeniach nie mierzono energii w dżulach ani nie wykazano przewagi nad Transformerami.

## Co jest dostępne?

Na razie publikujemy tylko ten opis. Nie ma tu jeszcze raportu, kodu, wag ani danych do odtworzenia eksperymentów. Neural Engine jest odrębnym projektem badawczym, nie oprogramowaniem produkcyjnego agenta M.A.R.I.A.
