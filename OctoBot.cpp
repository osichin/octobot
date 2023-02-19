#include <Windows.h>
#include <iostream>
#include <queue>
#include <iterator>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>
#include <ctime>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <dxgi1_2.h>3
#include <d3d11.h>

#pragma comment(lib, "D3D11.lib")


using namespace std;
using namespace cv;
using namespace cv::cuda;
using namespace std::chrono;

constexpr float CONFIDENCE_THRESHOLD    = 0;
constexpr float NMS_THRESHOLD           = 0.1;
constexpr int NUM_CLASSES               = 1;
constexpr int ACTIVATION_RANGE_W        = 832;
constexpr int ACTIVATION_RANGE_H        = 612;
constexpr bool DISPLAY                  = false;
constexpr bool AIMBOT                   = true;
constexpr bool MOUSE_CENTER             = true;

// colors for bounding boxes
const cv::Scalar colors[] = {
    {0, 255, 255},
    {255, 255, 0},
    {0, 255, 0},
    {255, 0, 0}
};
const auto NUM_COLORS = sizeof(colors) / sizeof(colors[0]);

std::tuple<int, int> get_screen_dimensions(HWND hwnd)
{
    // get the height and width of the screen
    RECT windowsize;
    GetClientRect(hwnd, &windowsize);
    return std::make_tuple(windowsize.bottom, windowsize.right);
}

std::tuple<int, int, int, int> get_viewport_dimensions(HWND hwnd, std::tuple<int, int> screen_dimensions)
{
    int viewport_height, viewport_width, viewport_x, viewport_y;

    viewport_x = get<1>(screen_dimensions) / 2 - ACTIVATION_RANGE_W / 2;
    viewport_y = get<0>(screen_dimensions) / 2 - ACTIVATION_RANGE_H / 2;
    viewport_height = ACTIVATION_RANGE_H;
    viewport_width = ACTIVATION_RANGE_W;

    return std::make_tuple(viewport_x, viewport_y, viewport_width, viewport_height);
}

std::tuple<float, float> get_bullseye(float box_x, float box_y, float box_w, float box_l)
{
    float x, y;
    x = (box_w / 2);
    y = (box_l / 7.0);

    return std::make_tuple(x, y);
}

std::tuple<float, float> get_mouse_coordinates(std::tuple<int, int> screen_dimensions)
{
    if (MOUSE_CENTER) {
        //std::cout << "x: " << get<0>(screen_dimensions) / 2 << "y" << get<1>(screen_dimensions) / 2 << std::endl;
        return std::make_tuple(get<1>(screen_dimensions) / 2, get<0>(screen_dimensions) / 2);

    }

    POINT p;
    float x, y = 0;
    if (GetCursorPos(&p))
    {
        //cursor position now in p.x and p.y
        x = p.x;
        y = p.y;
    }
    return std::make_tuple(x, y);
}

float get_distance(float x1, float y1, float x2, float y2)
{
    // Calculating distance 
    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2) * 1.0);
}

std::tuple<float, float> get_midpoint(int x1, int y1, int x2, int y2)
{
    // Calculating distance 
    return std::make_tuple((x1 + x2)/2.0, (y1 + y2)/2.0);
}

std::tuple<float, float> get_vector2target(int x1, int y1, float x2, float y2)
{
    return std::make_tuple(x2 - x1, y2 - y1);
}

float get_vector_norm(float x, float y)
{
    return sqrt(pow(x, 2) + pow(y, 2));
}

std::tuple<float, float> get_vector2target_unit(float x, float y, float norm)
{
    float rx, ry;

    if (x == 0) 
    {
        rx = 0;
    } 
    else 
    {
        rx = x / norm;
    }

    if (y == 0)
    {
        ry = 0;
    }
    else
    {
        ry = y / norm;
    }

    return std::make_tuple(rx, ry);
}


Mat hwnd2mat(HWND hwnd, tuple<int, int> screen_dimensions, tuple<int, int, int, int> viewport_dimensions) {

    HDC hwindowDC, hwindowCompatibleDC;

    int srcheight, srcwidth, viewport_x, viewport_y, viewport_width, viewport_height;
    HBITMAP hbwindow;
    Mat src;
    BITMAPINFOHEADER  bi;

    hwindowDC = GetDC(hwnd);
    hwindowCompatibleDC = CreateCompatibleDC(hwindowDC);
    SetStretchBltMode(hwindowCompatibleDC, COLORONCOLOR);

    srcheight       = get<0>(screen_dimensions);
    srcwidth        = get<1>(screen_dimensions);

    viewport_x      = get<0>(viewport_dimensions);
    viewport_y      = get<1>(viewport_dimensions);
    viewport_height = get<3>(viewport_dimensions);
    viewport_width  = get<2>(viewport_dimensions);

    src.create(viewport_height, viewport_width, CV_8UC4);

    // create a bitmap
    hbwindow = CreateCompatibleBitmap(hwindowDC, viewport_width, viewport_height);
    bi.biSize = sizeof(BITMAPINFOHEADER);    //http://msdn.microsoft.com/en-us/library/windows/window/dd183402%28v=vs.85%29.aspx
    bi.biWidth = viewport_width;
    bi.biHeight = -viewport_height;  //this is the line that makes it draw upside down or not
    bi.biPlanes = 1;
    bi.biBitCount = 32;
    bi.biCompression = BI_RGB;
    bi.biSizeImage = 0;
    bi.biXPelsPerMeter = 0;
    bi.biYPelsPerMeter = 0;
    bi.biClrUsed = 0;
    bi.biClrImportant = 0;

    // use the previously created device context with the bitmap
    SelectObject(hwindowCompatibleDC, hbwindow);
    // copy from the window device context to the bitmap device context
    StretchBlt(hwindowCompatibleDC, 0, 0, viewport_width, viewport_height, hwindowDC, viewport_x, viewport_y, viewport_width, viewport_height, SRCCOPY); //change SRCCOPY to NOTSRCCOPY for wacky colors !
    GetDIBits(hwindowCompatibleDC, hbwindow, 0, viewport_height, src.data, (BITMAPINFO*)&bi, DIB_RGB_COLORS);  //copy from hwindowCompatibleDC to hbwindow

    // avoid memory leak
    DeleteObject(hbwindow); DeleteDC(hwindowCompatibleDC); ReleaseDC(hwnd, hwindowDC);

    return src;
}

void MouseMove(int x, int y)
{
    INPUT Input = { 0 };
    ZeroMemory(&Input, sizeof(INPUT));
    Input.type = INPUT_MOUSE;
    Input.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE;
    Input.mi.dx = x * 0xFFFF / ::GetSystemMetrics(SM_CXSCREEN) + 1;
    Input.mi.dy = y * 0xFFFF / ::GetSystemMetrics(SM_CYSCREEN) + 1;
    Input.mi.dwExtraInfo = 123;
    ::SendInput(1, &Input, sizeof(INPUT));
}

void Release() {

    ClipCursor(NULL);
}


void Block() {
    RECT clipRect;

    clipRect.left = ::GetSystemMetrics(SM_CXSCREEN) / 2 ;
    clipRect.top = ::GetSystemMetrics(SM_CYSCREEN) / 2;
    clipRect.right = ::GetSystemMetrics(SM_CXSCREEN) / 2 ;
    clipRect.bottom = ::GetSystemMetrics(SM_CYSCREEN) / 2 ;

    ClipCursor(&clipRect);

}

void Fire()
{
    INPUT Input;
    Input.type = INPUT_KEYBOARD;
    Input.ki.wScan = 0x45; // hardware scan code for key
    Input.ki.time = 0;
    Input.ki.dwExtraInfo = 0;
    //Input.ki.wVk = 0; // virtual-key code for the "a" key
    Input.ki.dwFlags = 0; // 0 for key press

    SendInput(1, &Input, sizeof(INPUT));
    
    Input.ki.dwFlags = KEYEVENTF_KEYUP;
    SendInput(1, &Input, sizeof(INPUT));
}

bool is_mouse_pressed()
{
    return (((GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0) || ((GetAsyncKeyState(VK_LSHIFT) & 0x8000) != 0));
}

bool is_alt_pressed()
{
    return (GetAsyncKeyState(VK_MENU) & 0x8000);
}



ID3D11Device* _lDevice;
ID3D11DeviceContext* _lImmediateContext;
IDXGIOutputDuplication* _lDeskDupl;
ID3D11Texture2D* _lAcquiredDesktopImage;
DXGI_OUTPUT_DESC _lOutputDesc;
DXGI_OUTDUPL_DESC _lOutputDuplDesc;
D3D11_MAPPED_SUBRESOURCE _resource;
ID3D11Texture2D* currTexture;

bool init_desktop_dup() {
    // Feature levels supported
    D3D_FEATURE_LEVEL gFeatureLevels[] = {
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0,
        D3D_FEATURE_LEVEL_9_1
    };
    UINT gNumFeatureLevels = ARRAYSIZE(gFeatureLevels);
    D3D_FEATURE_LEVEL lFeatureLevel;

    HRESULT hr(E_FAIL);
    hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 
        D3D11_CREATE_DEVICE_FLAG::D3D11_CREATE_DEVICE_SINGLETHREADED, 
        gFeatureLevels, gNumFeatureLevels, D3D11_SDK_VERSION, 
        &_lDevice, &lFeatureLevel, &_lImmediateContext);

    if (FAILED(hr))
        return false;

    if (!_lDevice)
        return false;

    // Get DXGI device
    IDXGIDevice* lDxgiDevice;
    hr = _lDevice->QueryInterface(__uuidof(IDXGIDevice), reinterpret_cast<void**>(&lDxgiDevice));
    if (FAILED(hr))
        return false;

    // Get DXGI adapter
    IDXGIAdapter* lDxgiAdapter;
    hr = lDxgiDevice->GetParent(__uuidof(IDXGIAdapter), reinterpret_cast<void**>(&lDxgiAdapter));
    lDxgiDevice->Release();
    lDxgiDevice = nullptr;
    if (FAILED(hr))
        return false;

    UINT Output = 0;
    // Get output
    IDXGIOutput* lDxgiOutput;
    hr = lDxgiAdapter->EnumOutputs(Output, &lDxgiOutput);

    if (FAILED(hr))
        return false;
    lDxgiAdapter->Release();
    lDxgiAdapter = nullptr;

    hr = lDxgiOutput->GetDesc(&_lOutputDesc);

    if (FAILED(hr))
        return false;

    // QI for Output 1
    IDXGIOutput1* lDxgiOutput1;
    hr = lDxgiOutput->QueryInterface(__uuidof(lDxgiOutput1), reinterpret_cast<void**>(&lDxgiOutput1));
    lDxgiOutput->Release();
    lDxgiOutput = nullptr;
    if (FAILED(hr))
        return false;

    // Create desktop duplication
    hr = lDxgiOutput1->DuplicateOutput(_lDevice, &_lDeskDupl);

    if (FAILED(hr))
        return false;

    lDxgiOutput1->Release();
    lDxgiOutput1 = nullptr;

    // Create GUI drawing texture
    _lDeskDupl->GetDesc(&_lOutputDuplDesc);
    // Create CPU access texture
    D3D11_TEXTURE2D_DESC desc;
    desc.Width = _lOutputDuplDesc.ModeDesc.Width;
    desc.Height = _lOutputDuplDesc.ModeDesc.Height;
    desc.Format = _lOutputDuplDesc.ModeDesc.Format;
    desc.ArraySize = 1;
    desc.BindFlags = 0;
    desc.MiscFlags = 0;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.MipLevels = 1;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_FLAG::D3D11_CPU_ACCESS_READ;
    desc.Usage = D3D11_USAGE::D3D11_USAGE_STAGING;

    hr = _lDevice->CreateTexture2D(&desc, NULL, &currTexture);
    if (!currTexture)
    {
        hr = _lDeskDupl->ReleaseFrame();
        return false;
    }

    return true;
}

bool capture_screen()
{
    HRESULT hr(E_FAIL);
    IDXGIResource* lDesktopResource = nullptr;
    DXGI_OUTDUPL_FRAME_INFO lFrameInfo;

    hr = _lDeskDupl->AcquireNextFrame(999, &lFrameInfo, &lDesktopResource);

    if (FAILED(hr))
        return false;

    if (lFrameInfo.LastPresentTime.HighPart == 0) // not interested in just mouse updates, which can happen much faster than 60fps if you really shake the mouse
    {
        hr = _lDeskDupl->ReleaseFrame();
        return false;
    }

    // QI for ID3D11Texture2D
    hr = lDesktopResource->QueryInterface(__uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&_lAcquiredDesktopImage));
    lDesktopResource->Release();
    lDesktopResource = nullptr;
    if (FAILED(hr))
    {
        hr = _lDeskDupl->ReleaseFrame();
        return false;
    }

    _lImmediateContext->CopyResource(currTexture, _lAcquiredDesktopImage);
    UINT subresource = D3D11CalcSubresource(0, 0, 0);
    _lImmediateContext->Map(currTexture, subresource, D3D11_MAP_READ, 0, &_resource);
    _lImmediateContext->Unmap(currTexture, 0);
    hr = _lDeskDupl->ReleaseFrame();

    return true;
}


HHOOK hMouseHook;       // Low level mouse hook
bool mouse_blocked = false;

LRESULT CALLBACK LowLevelMouseProc(int nCode, WPARAM wParam, LPARAM lParam)
{
    
    MOUSEHOOKSTRUCT* pMouseStruct = (MOUSEHOOKSTRUCT*)lParam;
    cerr << "nCode: " << nCode << " wParam: " << wParam << " lParam: " << lParam << " mouse_block: " << mouse_blocked << " xtra: " << pMouseStruct->dwExtraInfo << endl;
    if (nCode == 0 && pMouseStruct != NULL && wParam == 512 && pMouseStruct->dwExtraInfo == 0 && mouse_blocked) {
        return -1;
    }
    else {

        return CallNextHookEx(hMouseHook, nCode, wParam, lParam);
    }

}

void LLHOOKThread() {
    hMouseHook = SetWindowsHookEx(WH_MOUSE_LL, &LowLevelMouseProc, GetModuleHandle(NULL), 0);

    
    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0) > 0)
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    UnhookWindowsHookEx(hMouseHook);
    
}


//int main(int argc, const char* argv[])
int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR pCmdLine, int nCmdShow)
{
    cout << getBuildInformation() << std::endl;
    cout << "OpenCV version : " << CV_VERSION << endl;
    cout << "Major version : " << CV_MAJOR_VERSION << endl;
    cout << "Minor version : " << CV_MINOR_VERSION << endl;
    //cout << "Subminor version : " << CV_SUBMINOR_VERSION << endl;

    //std::thread t1(LLHOOKThread);

    HWND hwndDesktop = GetDesktopWindow();
    auto screen_dimensions      = get_screen_dimensions(hwndDesktop);
    auto viewport_dimensions    = get_viewport_dimensions(hwndDesktop, screen_dimensions);
    if (DISPLAY) {
        namedWindow("output", WINDOW_NORMAL);
        resizeWindow("output", ACTIVATION_RANGE_W, ACTIVATION_RANGE_H);
    }

    auto net = cv::dnn::readNetFromDarknet("yolov4-obj.cfg", "yolov4-obj_best.weights");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    auto output_names = net.getUnconnectedOutLayersNames();

    cv::Mat frame, blob;
    cv::Rect crop_rect(get<0>(viewport_dimensions), get<1>(viewport_dimensions), get<2>(viewport_dimensions), get<3>(viewport_dimensions));
    std::vector<cv::Mat> detections;
    init_desktop_dup();

    Sleep(700);

    std::tuple<float, float> last_target = { 0, 0 };
    float last_target_x = get<0>(screen_dimensions) / 2;
    float last_target_y = get<1>(screen_dimensions) / 2;

    while (true)
    {


        //Mat frame = hwnd2mat(hwndDesktop, screen_dimensions, viewport_dimensions);

        // try another method here
        if (capture_screen()) {

            //high_resolution_clock::time_point t1 = high_resolution_clock::now();

            frame = cv::Mat(_lOutputDuplDesc.ModeDesc.Height, _lOutputDuplDesc.ModeDesc.Width, CV_8UC4, _resource.pData);
            frame = frame(crop_rect);
            cvtColor(frame, frame, COLOR_RGBA2RGB);


            cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(416, 416), cv::Scalar(), true, false, CV_32F);
            net.setInput(blob);
            net.forward(detections, output_names);
            
            std::vector<int> indices[NUM_CLASSES];
            std::vector<cv::Rect> boxes[NUM_CLASSES];
            std::vector<float> scores[NUM_CLASSES];

            for (auto& output : detections)
            {
                const auto num_boxes = output.rows;
                for (int i = 0; i < num_boxes; i++)
                {
                    auto x = output.at<float>(i, 0) * frame.cols;
                    auto y = output.at<float>(i, 1) * frame.rows;
                    auto width = output.at<float>(i, 2) * frame.cols;
                    auto height = output.at<float>(i, 3) * frame.rows;
                    cv::Rect rect(x - width / 2, y - height / 2, width, height);


                    auto confidence = *output.ptr<float>(i, 5);
                    if (confidence >= CONFIDENCE_THRESHOLD)
                    {
                        boxes[0].push_back(rect);
                        scores[0].push_back(confidence);
                    }
                }
            }

            cv::dnn::NMSBoxes(boxes[0], scores[0], 0.0, NMS_THRESHOLD, indices[0]);


            if (indices[0].size() > 0)
            {

                // go through all of the discovered boxes now
                float min_distance = 99999;
                float dist;
                tuple<int, int> mouse_coord = get_mouse_coordinates(screen_dimensions);
                int mouse_x = get<0>(mouse_coord);
                int mouse_y = get<1>(mouse_coord);
                int min_index = -1;

                for (size_t i = 0; i < indices[0].size(); ++i)
                {

                    // these coordinates are relative to the viewport box!
                    const auto& rect = boxes[0][indices[0][i]];
                    cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), colors[0], 2);
                    tuple<float, float> bulls_eye = get_bullseye(rect.x, rect.y, rect.width, rect.height);

                    dist = get_distance((rect.x + get<0>(bulls_eye) + get<0>(viewport_dimensions)), (rect.y + get<1>(bulls_eye) + get<1>(viewport_dimensions)),
                        (mouse_x), (mouse_y));

                    if (dist < min_distance)
                    {
                        min_distance = dist;
                        min_index = i;
                    }
                }

                auto& target_box = boxes[0][indices[0][min_index]];
                tuple<float, float> bulls_eye = get_bullseye(target_box.x, target_box.y, target_box.width, target_box.height);
                cv::rectangle(frame, cv::Point(target_box.x, target_box.y), cv::Point(target_box.x + target_box.width, target_box.y + target_box.height), colors[1], 2);
                cv::circle(frame, cv::Point((target_box.x + get<0>(bulls_eye)), (target_box.y + get<1>(bulls_eye))), 5, cv::Scalar(0, 0, 255), -1);


                float target_x = target_box.x + get<0>(bulls_eye) + get<0>(viewport_dimensions);
                float target_y = target_box.y + get<1>(bulls_eye) + get<1>(viewport_dimensions);

                std::tuple<float, float> midpoint = get_midpoint(target_x, target_y, mouse_x, mouse_y);

                // the true target is half way in between because as we aim toward the target, it moves toward us
                float true_target_x = target_x; // get<0>(midpoint) 
                float true_target_y = target_y; // get<1>(midpoint)

                std::tuple<float, float> vector2target = get_vector2target(mouse_x, mouse_y, true_target_x, true_target_y);

                float vector_norm = get_vector_norm(get<0>(vector2target), get<1>(vector2target));
                
                std::tuple<float, float> vector2target_unit = get_vector2target_unit(get<0>(vector2target), get<1>(vector2target), vector_norm);
                
                float distance = get_distance(mouse_x, mouse_y, true_target_x, true_target_y);
                
                //distance = distance;3
                float AIM_SPEED = distance * 0.5;

                //if (distance >= 5) {
                //    AIM_SPEED = distance / 3;
                //} else if (distance >= 10) {
                //    AIM_SPEED = distance / 4;
                //}else if (distance >= 50) {
                //    AIM_SPEED = distance / 5;
                //}

               
                //std::cout << distance << std::endl;
                float aim_vector_x = get<0>(vector2target_unit) * AIM_SPEED;
                float aim_vector_y = get<1>(vector2target_unit) * AIM_SPEED;
                //std::cout << "distance: " << di3stance << std::endl;
                // at this point we know exactly where we are aiming on our target :D
                if (is_mouse_pressed() && AIMBOT && !is_alt_pressed())
                {
                    //mouse_blocked = true;
                    //Block();
                    //float dist_to_last = get_distance(last_target_x, last_target_y, true_target_x, true_target_y);
                    //if (dist_to_last >= 5) {
                    if (distance <= 5) {
                        Fire();
                    }
                        MouseMove(mouse_x + aim_vector_x, mouse_y + aim_vector_y);
                        //std::cout << "distance: " << distance << " x: " << aim_vector_x << " y: " << aim_vector_y <<  std::endl;

     
                       // Release();
                        
                    //}

                    //last_target_x = true_target_x;
                    //last_target_y = true_target_y;
                }
                else {
                    //mouse_blocked = false;
                }
               
            }
            else {
                //mouse_blocked = false;
            }

            //high_resolution_clock::time_point t2 = high_resolution_clock::now();
            //duration<double, std::milli> time_span = t2 - t1;
            //std::cout << time_span.count() << std::endl;


            // you can do some image processing here
            if (DISPLAY) {
                imshow("output", frame);
                cv::waitKey(1);
            }

        }


    }

}