from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch

# Initialize tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased-distilled-squad')
model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-cased-distilled-squad')

# Define question and context
question, text = "Who was Jim Henson?", "Jim Henson was a nice puppeteer."

# Tokenize input (question and text/context)
inputs = tokenizer(question, text, return_tensors="pt")

# Predict answer spans using the model
with torch.no_grad():
    outputs = model(**inputs)

# Retrieve the most likely start and end of answer indices
answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits
answer_start = torch.argmax(answer_start_scores)
answer_end = torch.argmax(answer_end_scores) + 1  # Add 1 to include the end index in the slice

# Convert indices to tokens, then tokens to string to get the answer
answer_tokens = inputs.input_ids[0, answer_start:answer_end]
answer = tokenizer.decode(answer_tokens)

print(answer)



question = "Can the ceiling increase or decrease during project implementation?"
text = f"""
burping in the project. Note: If the project design requires it, the disbursement percentage for one or more items reaching 100 percent is possible provided that the entire cost of the Bank's project is within the framework of the Bank's specifications. amount of national funding for cost sharing. 41. For projects in countries that have not yet adopted country financing parameters or have not yet adopted Operational Policy 6.00, see appendix I. 42. The Bank may require the borrower to implement certain activities before withdrawing capital from the loan. These conditions are specified in the legal agreement. Expenditures Expenditure categories Disbursement rates Disbursement conditions 14151415 expenditures governed by such conditions will be clearly separated from the remaining expenditures in the project and classified as separate expenditures, so the Bank can check and monitor things
sue carefully. The following are a few examples of situations where disbursement conditions on a portion of a loan are common:"""

# Tokenize input (question and text/context)
inputs = tokenizer(question, text, return_tensors="pt")

# Predict answer spans using the model
with torch.no_grad():
    outputs = model(**inputs)

# Retrieve the most likely start and end of answer indices
answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits
answer_start = torch.argmax(answer_start_scores)
answer_end = torch.argmax(answer_end_scores) + 1  # Add 1 to include the end index in the slice

# Convert indices to tokens, then tokens to string to get the answer
answer_tokens = inputs.input_ids[0, answer_start:answer_end]
answer = tokenizer.decode(answer_tokens)

print(answer)

# question = "Tạm ứng dư thừa/khoản dư thừa là gì ?"
# text = f"""
# thanh toán.  •  Tạm ứng: là khoản thanh toán cho bên vay các khoản chi phí của dự án đã được  dự trù trước. Bên vay, theo định kỳ nộp các chứng từ cho thấy rằng các chi phí đã  phát sinh và được chi trả từ tiền được tạm ứng Giải ngân cho các bên thứ ba •      Thanh toán trực tiếp: Thanh toán cho bên thứ ba (như nhà thầu, nhà cung  cấp, tư vấn) đối với chi phí cho các khoản chi tiêu của dự án. Bên vay cung cấp  chứng từ chứng minh rằng các khoản chi tiêu như vậy đã được thực hiện vào thời  điểm có yêu cầu thanh toán cho bên thứ ba.  • Cam kết đặc biệt: Thanh toán cho một tổ chức tài chính đối với chi phí cho các  khoản chi tiêu của dự án được thực hiện theo một cam kết đặc biệt. Một cam kết  đặc biệt là một cam kết không hủy ngang 
# với sự tham gia của Ngân hàng bằng  văn bản cam kết thanh toán các khoản đó kể cả khi bị đình chỉ hoặc hủy vốn  nào sau đó. Tổ chức tài chính đưa ra xác nhận rằng các khoản chi tiêu như vậy đã  được thực hiện vào thời điểm được yêu cầu thanh toán. 18191819 kế hoạch mua sắm đấu thầu, và kinh nghiệm giải ngân của  bên vay với Ngân hàng cũng được đưa vào xem xét cùng với  các vấn đề sau: •  bản chất và số lượng các khoản chi tiêu dự kiến của dự án  và các yêu cầu của các nhà cung cấp mà có thể thấy trước  được, •  kế hoạch tài chính tổng thể và khả năng tài trợ đối ứng  kịp thời của bên vay, •  mức độ tập trung hóa phân tán của người hưởng lợi của  dự án, •  khả năng tài trợ trước của bên vay cho các khoản chi  tiêu, •  yêu cầu định 
# kỳ về tiền mặt của dự án, •  các bước chuẩn bị báo cáo tài chính. Chứng từ Hỗ trợ 57. Ngân hàng yêu cầu chứng từ hỗ trợ mà các chứng từ  này cung cấp được bằng chứng cho thấy việc rút vốn từ tài  khoản vay đã được hoặc đang được dành cho các khoản chi  tiêu hợp lệ, như được định rõ trong hiệp định pháp lý và  phần 4 của Hướng dẫn Giải ngân. Tùy theo phương pháp  giải ngân được sử dụng, bên vay có thể cung cấp chứng từ  này cùng với lúc nộp 
# đơn xin rút vốn hoặc vào một ngày  sau đó (xem phần về “Việc sử dụng Phương pháp tạm ứng”  trong chương V). 58. Chứng từ hỗ trợ có thể dưới dạng (a) copy của chứng từ  gốc chứng minh rằng thanh toán đã được thực hiện hoặc  đến hạn phải trả cho các chi tiêu hợp lệ (ví dụ hóa đơn
# biệt).  Tỉ lệ giải ngân: là tỉ lệ phần trăm các chi tiêu  hợp lệ được cấp vốn theo một dự án.  Quyết toán Chứng từ: là thuật ngữ chung  được sử dụng để đưa ra bằng chứng hỗ trợ cho  một quyết định hay hành động có được thực  hiện hay không. Trong ngữ cảnh kế toán, thông  thường, quyết toán chứng từ cung cấp các tài  liệu hỗ trợ chẳng hạn như các chứng từ, hoá đơn  của bên bán, chấp nhận thanh toán, thông báo  vận chuyển, vv.) hoặc báo cáo chi phí tóm tắt để  hỗ trợ cho việc thanh toán, hoàn trả, nhập bút  toán kế toán, hoặc sự kiện kế toán khác (xem  định nghĩa về chứng từ, tài liệu hỗ trợ, và báo  cáo tóm tắt). 62636263 PHỤ CHƯƠNG A - ĐỊNH NGHĨA CÁC THUẬT NGỮ  Ngày hiệu lực: là ngày mà Ngân hàng thông  báo cho bên vay Ngân hàng chấp nhận bằng  chứng yêu cầu để chứng minh các điều kiện hiệu  quả theo hiệp định pháp lý đã được đáp ứng, và  là ngày hiệp định pháp lý có hiệu lực. Việc rút  vốn có thể thực hiện từ tài khoản vay kể từ ngày  này, chẳng hạn như, bẳt đầu giải ngân.  Chi tiêu hợp lệ: là chi tiêu hợp lệ, theo hiệp  định pháp lý, sẽ được chi trả từ tiền của khoản  vay.    Tạm ứng dư thừa/khoản dư thừa: là khoản  tiền được gửi vào tài khoản chuyên dùng mà,  trong thời gian tới,  không cần thiết cho việc chi  trả các khoản thanh toán những chi tiêu hợp lệ.    Hạng mục chi tiêu: là hạng mục chi tiêu hợp  lệ 
# có thể được tài trợ từ tiền của khoản vay.    Tổ chức tài chính: là ngân hàng thương mại,  ngân hàng trung ương, hoặc các tổ chức khác  đáp ứng được các tiêu trí trong Hướng dẫn Giải  ngân để nắm giữ tài khoản chuyên dùng.  Khoản tự làm: là các công trình dân dụng  được  thực  hiện  bởi  cơ  quan  chính  phủ  địa  phương của bên vay sử dụng lực lượng lao động  của riêng họ.  Phí ban đầu: là khoản phí IBRD tính phí các  bên vay về khoản vay, khoản phí này phải thanh  toán vào ngày hiệu lực của khoản vay, và theo  quyền tự quyết của bên vay, cũng như được qui  định cụ thể trong hiệp định pháp lý, có thể được  chi trả từ tiền vay.  Điều kiện Chung đối với các khoản Tín  dụng và Tài trợ: là những điều khoản và điều  kiện chuẩn được đưa ra để áp dụng cho các hiệp  định tài trợ giữa bên nhận tài trợ và IDA, theo các  điều khoản của hiệp định nà
# t toán (bằng chữ) 16. Nếu chứng từ chung cho nhiều khoản vay (theo mục 2 ở trên), điền số tiền phân bổ cho từng khoản tài trợ      Số tham chiếu khoản vay/tài trợ/viện trợ             Số tiền                        Số tham chiếu khoản vay/tài trợ/viện trợ                                Số tiền 14. Hạng mục và số tham chiếu hợp đồng - nếu chi tiêu liên quan  tới nhiều hạng mục hay tham chiếu hợp đồng thì để trống mục 14a và  14b và bổ sung thông tin này vào trong chứng từ kèm theo 14a. Tham chiếu hạng mục   14b. Tham chiếu hợp đồng D. Xác nhận và ký tên Người ký tên dưới đây xác nhận và đảm bảo những điều sau: A. Nếu người ký tên yêu cầu Tạm ứng vào tài khoản chuyên dùng: (1) số tiền yêu cầu tương ứng với chi phí dự kiến của dự án 
# đã trình cho Ngân hàng thế giới; và (2) chứng từ của khoản ứng trước này (sẽ)  được chuyển cho Ngân hàng thế giới theo kỳ báo cáo nêu trong các hiệp định  pháp lý liên quan hoặc thư giải ngân của dự án. Nếu người ký tên dưới đây quyết toán các thanh toán từ tài khoản chuyên  dùng: (a) chi tiêu nêu trong đơn hợp lệ được tài trợ từ khoản vay/tài trợ/viện trợ theo các điều khoản của (các) hiệp định có tính pháp lý, và (b) người ký tên dưới đây trước đây chưa và sẽ không có ý định  tìm nguồn tài chính khác để sử dụng cho những khoản chi tiêu này. B. Nếu người ký tên dưới đây yêu cầu hoàn vốn hay thanh toán trực tiếp: (1) những chi tiêu nêu trong đơn thuộc đối tượng cấp kinh phí từ các đợt giải ngân của khoản vay/tài trợ/viện trợ theo 
# các điều khoản  của (các) văn bản pháp lý liên quan; và (b) người ký tên dưới đây chưa bao giờ và sẽ không có ý định tìm nguồn tài chính khác để sử 
# dụng cho những khoản chi này. C. Nếu người ký tên dưới đây yêu cầu giải ngân đợt cấp vốn trong phạm vi khoản vay chính sách phát triển/tài trợ/viện 
# trợ: (1) số tiền tương đương với số tiền ký quỹ  sẽ được chịu trách nhiệm kế toán theo các  điều khoản của (các) văn bản pháp lý liên quan, và (2) sẽ không sử dụng nguồn vốn của khoản vay/tài trợ/viện trợ để dùng cho chi tiêu không hợp lệ nêu trong các văn bản pháp lý liên quan.Người ký tên  dưới đây sẽ lưu giữ toàn bộ chứng từ các chi tiêu thuộc Đơn xin rút vốn này để cung cấp cho bên kiểm toán và cán bộ kiểm tra của Ngân hàng thế giới. 17. Tê
# """

# inputs = tokenizer(question, text, return_tensors="pt")
# with torch.no_grad():
#     outputs = model(**inputs)

# print(outputs)



