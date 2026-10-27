; Inno Setup script for the OwlFluids Windows installer.
; Build after PyInstaller (packaging/build.ps1 does both):
;   iscc /DAppVersion=0.2.0 packaging\owlfluids.iss
; Output: dist\OwlFluids-Setup-<version>.exe
;
; Installs per user (no admin prompt) into %LOCALAPPDATA%\Programs\OwlFluids, adds
; that folder to the user's PATH so `owlfluids` works in any new terminal, and adds
; Start menu entries. Running a newer installer over an older one upgrades in place;
; the app's updater runs it with /SILENT, and it reopens the app when it finishes.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{0D96D3BD-FDEE-46E3-B50A-877B0512C787}
AppName=OwlFluids
AppVersion={#AppVersion}
AppVerName=OwlFluids {#AppVersion}
AppPublisher=OwlFluids
DefaultDirName={autopf}\OwlFluids
DefaultGroupName=OwlFluids
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist
OutputBaseFilename=OwlFluids-Setup-{#AppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ChangesEnvironment=yes
CloseApplications=yes
RestartApplications=no
UninstallDisplayIcon={app}\owlfluids-app.exe
#if FileExists(AddBackslash(SourcePath) + "owlfluids.ico")
SetupIconFile=owlfluids.ico
#endif

[Tasks]
Name: "addtopath"; Description: "Add OwlFluids to PATH (use ""owlfluids"" from any terminal)"
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[InstallDelete]
; drop the previous version's libraries so nothing stale is left behind after an upgrade
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "..\dist\OwlFluids\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\examples\*"; DestDir: "{app}\examples"; Flags: ignoreversion recursesubdirs
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\OwlFluids"; Filename: "{app}\owlfluids-app.exe"
Name: "{group}\OwlFluids examples"; Filename: "{app}\examples"
Name: "{group}\Uninstall OwlFluids"; Filename: "{uninstallexe}"
Name: "{autodesktop}\OwlFluids"; Filename: "{app}\owlfluids-app.exe"; Tasks: desktopicon

[Run]
; compile the solver for this PC now, under the progress bar, rather than on the first run
; (every version starts with a cold numba cache)
Filename: "{app}\owlfluids.exe"; Parameters: "run lox-feed --duration 0.1 --rate 0 --print-every 0"; Flags: runhidden waituntilterminated; StatusMsg: "Compiling the solver for this PC..."
; no skipifsilent: a silent upgrade from the app's updater reopens the app
Filename: "{app}\owlfluids-app.exe"; Description: "Open OwlFluids"; Flags: nowait postinstall

[Code]
const
  EnvKey = 'Environment';

function PathHas(Paths, Dir: string): Boolean;
begin
  Result := Pos(';' + Uppercase(Dir) + ';', ';' + Uppercase(Paths) + ';') > 0;
end;

procedure AddToPath(Dir: string);
var
  Paths: string;
begin
  if not RegQueryStringValue(HKCU, EnvKey, 'Path', Paths) then
    Paths := '';
  if PathHas(Paths, Dir) then
    exit;
  if (Paths <> '') and (Copy(Paths, Length(Paths), 1) <> ';') then
    Paths := Paths + ';';
  RegWriteExpandStringValue(HKCU, EnvKey, 'Path', Paths + Dir);
end;

procedure RemoveFromPath(Dir: string);
var
  Paths: string;
  P: Integer;
begin
  if not RegQueryStringValue(HKCU, EnvKey, 'Path', Paths) then
    exit;
  Paths := ';' + Paths + ';';
  P := Pos(';' + Uppercase(Dir) + ';', Uppercase(Paths));
  if P = 0 then
    exit;
  Delete(Paths, P, Length(Dir) + 1);
  Paths := Copy(Paths, 2, Length(Paths) - 2);
  RegWriteExpandStringValue(HKCU, EnvKey, 'Path', Paths);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep = ssPostInstall) and WizardIsTaskSelected('addtopath') then
    AddToPath(ExpandConstant('{app}'));
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    RemoveFromPath(ExpandConstant('{app}'));
end;
